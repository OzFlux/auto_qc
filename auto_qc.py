import copy
import datetime
import logging
import os
import sys

from configobj import ConfigObj
from fpdf import FPDF
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy
from numpy.lib.stride_tricks import sliding_window_view
import pytz

now = datetime.datetime.now()
log_file_base = "pfp_" + now.strftime("%Y%m%dT%H%M%S%f")
os.environ["pfp_log"] = log_file_base

sys.path.append("/home/peter-isaac/PyFluxPro")
from scripts import pysolar
from scripts import pfp_ck
from scripts import pfp_io
from scripts import pfp_utils

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout)
logger = logging.getLogger(log_file_base)
logger.setLevel(logging.DEBUG)

def init_info():
    d = {"messages": {"INFO": [], "WARNING": [], "ERROR": []},
         "to_pdf": {"messages": [], "images": []},
         "results": {},
         "passed": True}
    return d
def check_timestep(ds, info):
    info["check_timestep"] = init_info()
    iycts = info["check_timestep"]
    messages = iycts["messages"]
    #results = iycts["results"]
    to_pdf = iycts["to_pdf"]
    has_gaps = pfp_utils.CheckTimeStep(ds)
    if has_gaps:
        info["check_timestep"]["passed"] = False
        msg = "check_timestep: gaps or duplicates found in timestamp"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_timestep: passed timestamp check"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return info["check_timestep"]["passed"]
def check_required_variables(cfg, ds, info):
    info["check_required_variables"] = init_info()
    iycrv = info["check_required_variables"]
    messages = iycrv["messages"]
    to_pdf = iycrv["to_pdf"]
    missing_required_variables = []
    ds_labels = list(ds.root["Variables"].keys())
    required_variables = list(cfg["Variables"].keys())
    for required_variable in required_variables:
        if required_variable not in ds_labels:
            missing_required_variables.append(required_variable)
    if len(missing_required_variables) > 0:
        info["check_required_variables"]["passed"] = False
        msg = "check_required_variables: required variables missing "
        msg += ",".join(missing_required_variables)
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_required_variables: all required variables present"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return info["check_required_variables"]["passed"]
def check_percent_good(cfg, ds, info):
    info["check_percent_good"] = init_info()
    iycpg = info["check_percent_good"]
    messages = iycpg["messages"]
    results = iycpg["results"]
    results["failed"] = {"label": [], "percent_good": []}
    to_pdf = iycpg["to_pdf"]
    required_percentage_good = float(cfg["Options"]["required_percent_good"])
    ds_labels = list(ds.root["Variables"].keys())
    required_labels = list(cfg["Variables"].keys())
    for required_label in required_labels:
        if required_label in ds_labels:
            results[required_label] = {}
            var = pfp_utils.GetVariable(ds, required_label)
            percent_good = round(100*numpy.ma.count(var["Data"])/len(var["Data"]), 1)
            results[required_label]["percent_good"] = percent_good
            if percent_good < required_percentage_good:
                info["check_percent_good"]["passed"] = False
                results["failed"]["label"].append(required_label)
                results["failed"]["percent_good"].append(percent_good)
    if len(results["failed"]["label"]) > 0:
        info["check_percent_good"]["passed"] = False
        msg = "check_percent_good: "
        msg += ",".join(results["failed"]["label"]) + " failed"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_percent_good: all required variables passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return info["check_percent_good"]["passed"]
def check_outliers_ranges(cfg, ds, info):
    info["check_outliers_ranges"] = init_info()
    iycor = info["check_outliers_ranges"]
    iycor["passed"] = True
    messages = iycor["messages"]
    results = iycor["results"]
    results["outliers"] = {}
    to_pdf = iycor["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    required_labels = list(cfg["Variables"].keys())
    for required_label in required_labels:
        if required_label in ds_labels:
            var = pfp_utils.GetVariable(ds, required_label)
            upper = float(cfg["Variables"][required_label]["RangeCheck"]["upper"])
            lower = float(cfg["Variables"][required_label]["RangeCheck"]["lower"])
            idx = numpy.ma.where((var["Data"] > upper) | (var["Data"] < lower))[0]
            if len(idx) > 0:
                iycor["passed"] = False
                if required_label not in list(results["outliers"].keys()):
                    results["outliers"][required_label] = {"number": 0, "dates": [], "values": []}
                results["outliers"][required_label]["number"] = len(idx)
                results["outliers"][required_label]["dates"] = var["DateTime"][idx]
                results["outliers"][required_label]["values"] = var["Data"][idx]
                results["outliers"][required_label]["limits"] = ",".join([str(lower), str(upper)])
    if len(list(results["outliers"].keys())) > 0:
        msg = "check_outliers_ranges: "
        msg += ",".join(list(results["outliers"].keys())) + " have outliers"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
        for label in list(results["outliers"].keys()):
            limits = results["outliers"][label]["limits"]
            number = str(results["outliers"][label]["number"])
            msg = "**** " + label + " (" + limits+"): " + number
            messages["ERROR"].append(msg)
            to_pdf["messages"].append(msg)
            logger.info(msg)
    else:
        msg = "check_outliers: all required variables passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return iycor["passed"]
def check_outliers_gmad(cfg, ds, info):
    """ Check for noise spikes using MAD filter recommended by Gemini."""
    info["check_outliers_gmad"] = init_info()
    iycog = info["check_outliers_gmad"]
    iycog["passed"] = True
    messages = iycog["messages"]
    results = iycog["results"]
    results["outliers"] = {}
    to_pdf = iycog["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    required_labels = list(cfg["Variables"].keys())
    # don't apply GMAD to fluxes, wind direction or fiction velocity
    for label in ["Fco2", "Fe", "Fh", "Wd", "ustar"]:
        if label in required_labels:
            required_labels.remove(label)
    for required_label in required_labels:
        if required_label in ds_labels:
            var = pfp_utils.GetVariable(ds, required_label)
            #nrecs = int(ds.root["Attributes"]["nc_nrecs"])
            #scale_factor = 1.4826
            scale_factor = 5
            #n_sigmas=5
            n_sigmas = 7
            window_size = 25
            if window_size % 2 == 0:
                window_size += 1
            pad_width = window_size // 2
            padded = pad_masked_array(var, pad_width)
            fill_val = numpy.ma.median(var["Data"])
            work_data = padded.filled(fill_val)
            windows = sliding_window_view(work_data, window_size)
            rolling_median = numpy.median(windows, axis=1)
            abs_deviations = numpy.abs(windows - rolling_median[:, numpy.newaxis])
            rolling_mad = numpy.median(abs_deviations, axis=1)
            threshold = n_sigmas * scale_factor * rolling_mad
            difference = numpy.abs(var["Data"] - rolling_median)
            is_spike = difference > threshold
            #is_zero_drop = (Sws["Data"] < 0.001) & (rolling_median > 0.01)
            #new_mask = numpy.ma.mask_or(Sws["Data"].mask, (is_spike | is_zero_drop))
            #new_mask = numpy.ma.mask_or(var["Data"].mask, is_spike)
            idx = numpy.where(is_spike == True)[0]
            if len(idx) > 0:
                iycog["passed"] = False
                if required_label not in list(results["outliers"].keys()):
                    results["outliers"][required_label] = {}
                results["outliers"][required_label]["number"] = len(idx)
                results["outliers"][required_label]["dates"] = var["DateTime"][idx]
                results["outliers"][required_label]["values"] = var["Data"][idx]
                results["outliers"][required_label]["indices"] = idx
    if len(list(results["outliers"].keys())) > 0:
        msg = "check_outliers_gmad: "
        msg += ",".join(list(results["outliers"].keys())) + " have outliers"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
        for label in list(results["outliers"].keys()):
            number = results["outliers"][label]["number"]
            msg = "**** " + label + ": GMAD spikes " + str(number)
            messages["ERROR"].append(msg)
            to_pdf["messages"].append(msg)
            logger.info(msg)
    else:
        msg = "check_outliers_gmad: all required variables passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return iycog["passed"]
def check_outliers_imad(cfg, ds, info):
    """ Check for noise spikes using the ICOS MAD filter."""
    info["check_outliers_imad"] = init_info()
    iycoi = info["check_outliers_imad"]
    iycoi["passed"] = True
    messages = iycoi["messages"]
    results = iycoi["results"]
    results["outliers"] = {}
    to_pdf = iycoi["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    required_labels = list(cfg["Variables"].keys())
    for required_label in required_labels:
        if required_label in ds_labels:
            pfp_ck.do_madfilter(cfg, ds, "Variables", required_label, code=24)
            var = pfp_utils.GetVariable(ds, required_label)
            idx = numpy.where(var["Flag"] == 24)[0]
            if len(idx) > 0:
                iycoi["passed"] = False
                if required_label not in list(results["outliers"].keys()):
                    results["outliers"][required_label] = {}
                results["outliers"][required_label]["number"] = len(idx)
                results["outliers"][required_label]["dates"] = var["DateTime"][idx]
                results["outliers"][required_label]["values"] = var["Data"][idx]
                results["outliers"][required_label]["indices"] = idx
    if len(list(results["outliers"].keys())) > 0:
        msg = "check_outliers_imad: "
        msg += ",".join(list(results["outliers"].keys())) + " have outliers"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
        for label in list(results["outliers"].keys()):
            number = results["outliers"][label]["number"]
            msg = "**** " + label + ": IMAD spikes " + str(number)
            messages["ERROR"].append(msg)
            to_pdf["messages"].append(msg)
            logger.info(msg)
    else:
        msg = "check_outliers_imad: all required variables passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return iycoi["passed"]
def check_radiation(cfg, ds, info):
    info["check_radiation"] = init_info()
    iycs = info["check_radiation"]
    messages = iycs["messages"]
    results = iycs["results"]
    to_pdf = iycs["to_pdf"]
    #nrecs = int(ds.root["Attributes"]["nc_nrecs"])
    Fsd = pfp_utils.GetVariable(ds, "Fsd")
    Fsd_toa = pfp_utils.GetVariable(ds, "Fsd_toa")
    Fsu = pfp_utils.GetVariable(ds, "Fsu")
    ldt = pfp_utils.GetVariable(ds, "DateTime")
    day_night = pfp_utils.GetVariable(ds, "day_night")
    # check the daytime Fsd is less than the TOA Fsd
    dgtidx = numpy.ma.where((Fsd["Data"] > 1.10*Fsd_toa["Data"]) & (day_night["Data"] == 1))[0]
    dgtidx = get_contiguous_regions(ds, dgtidx, 6)
    if len(dgtidx) > 0:
        info["check_radiation"]["passed"] = False
        results["dgtidx"] = dgtidx
        msg = "check_radiation: Fsd[day] > 1.1*TOA "+str(len(dgtidx))+" times"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_radiation: no Fsd[day] >  1.10*Fsd[TOA] detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    # check if the night time Fsd is < -10 W/m^2
    nltidx = numpy.ma.where((Fsd["Data"] < -10) & (day_night["Data"] == 0))[0]
    nltidx = get_contiguous_regions(ds, nltidx, 6)
    if len(nltidx) > 0:
        info["check_radiation"]["passed"] = False
        results["nltidx"] = nltidx
        msg = "check_radiation: Fsd[night] < -10 "+str(len(nltidx))+" times"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_radiation: no Fsd[night] < -10 W/m^2 detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    # check if the night time Fsd is > 10 W/m^2
    ngtidx = numpy.ma.where((Fsd["Data"] > 10) & (day_night["Data"] == 0))[0]
    ngtidx = get_contiguous_regions(ds, ngtidx, 6)
    if len(ngtidx) > 0:
        info["check_radiation"]["passed"] = False
        results["ngtidx"] = ngtidx
        msg = "check_radiation: Fsd[night] > 10 "+str(len(ngtidx))+" times"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_radiation: no Fsd[night] > 10 W/m^2 detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    # check daytime (1000 to 1500) albedo
    al = Fsu["Data"]/Fsd["Data"]
    hm = numpy.array([dt.hour+dt.minute/60 for dt in ldt["Data"]])
    dalgtidx = numpy.ma.where((al > 0.3) & ((hm > 10) & (hm < 15)))[0]
    if len(dalgtidx) > 0:
        info["check_radiation"]["passed"] = False
        results["dalgtidx"] = dalgtidx
        msg = "check_radiation: Fsu[day]/Fsd[day] > 0.3 "+str(len(dalgtidx))+" times"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_radiation: no albedo[day] > 0.3 detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    dalltidx = numpy.ma.where((al < 0.05) & ((hm > 10) & (hm < 15)))[0]
    if len(dalltidx) > 0:
        info["check_radiation"]["passed"] = False
        results["dalltidx"] = dalltidx
        msg = "check_radiation: Fsu[day]/Fsd[day] < 0.05 "+str(len(dalltidx))+" times"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_radiation: no albedo[day] < 0.05 detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    Fld = pfp_utils.GetVariable(ds, "Fld")
    Flu = pfp_utils.GetVariable(ds, "Flu")
    lwave_diff = Flu["Data"] - Fld["Data"]
    fldgtflu = numpy.ma.where(lwave_diff < -10)[0]
    if len(fldgtflu) > 0:
        info["check_radiation"]["passed"] = False
        results["fldgtflu"] = fldgtflu
        msg = "check_radiation: Flu < Fld-10 " + str(len(fldgtflu)) + " times"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_radiation: no Flu < Fld-10 detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return info["check_radiation"]["passed"]
def check_step_change_jump(cfg, ds, info):
    info["check_step_change_jump"] = init_info()
    iycscj = info["check_step_change_jump"]
    iycscj["passed"] = True
    messages = iycscj["messages"]
    results = iycscj["results"]
    results["jumps"] = {}
    to_pdf = iycscj["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    required_labels = list(cfg["Variables"].keys())
    for required_label in required_labels:
        if required_label not in ds_labels:
            msg = required_label + " not found in data structure"
            logger.info(msg)
            continue
        if "StepCheck" not in cfg["Variables"][required_label]:
            continue
        if "jump" not in cfg["Variables"][required_label]["StepCheck"]:
            continue
        jump = float(cfg["Variables"][required_label]["StepCheck"]["jump"])
        var = pfp_utils.GetVariable(ds, required_label)
        var["Diff"] = numpy.ma.ediff1d(var["Data"], to_begin=0)
        idx = numpy.ma.where(abs(var["Diff"]) > jump)[0]
        if len(idx) > 0:
            iycscj["passed"] = False
            if required_label not in list(results["jumps"].keys()):
                results["jumps"][required_label] = {}
            results["jumps"][required_label]["jump"] = jump
            results["jumps"][required_label]["number"] = len(idx)
            results["jumps"][required_label]["dates"] = var["DateTime"][idx]
            results["jumps"][required_label]["values"] = var["Data"][idx]
            results["jumps"][required_label]["diffs"] = var["Diff"][idx]
            results["jumps"][required_label]["indices"] = idx.copy()
    if len(list(results["jumps"].keys())) > 0:
        msg = "check_step_change_jump: "
        msg += ",".join(list(results["jumps"].keys())) + " have large jumps"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
        for label in list(results["jumps"].keys()):
            jump = results["jumps"][label]["jump"]
            number = results["jumps"][label]["number"]
            msg = "**** " + label + ": " + str(number) + " jumps > " + str(jump)
            messages["ERROR"].append(msg)
            to_pdf["messages"].append(msg)
            logger.info(msg)
    else:
        msg = "check_step_change_jump: all required variables passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return iycscj["passed"]
def check_step_change_zscore(cfg, ds, info):
    info["check_step_change_zscore"] = init_info()
    iycscz = info["check_step_change_zscore"]
    iycscz["passed"] = True
    messages = iycscz["messages"]
    results = iycscz["results"]
    results["jumps"] = {}
    to_pdf = iycscz["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    required_labels = list(cfg["Variables"].keys())
    for required_label in required_labels:
        if required_label not in ds_labels:
            msg = required_label + " not found in data structure"
            logger.info(msg)
            continue
        if "StepCheck" not in cfg["Variables"][required_label]:
            continue
        if "zfc" not in cfg["Variables"][required_label]["StepCheck"]:
            continue
        zfc = float(cfg["Variables"][required_label]["StepCheck"]["zfc"])
        var = pfp_utils.GetVariable(ds, required_label)
        var["Diff"] = numpy.ma.ediff1d(var["Data"], to_begin=0)
        mean_diff = numpy.ma.mean(var["Diff"])
        std_diff = numpy.ma.std(var["Diff"])
        idx = numpy.ma.where(abs(var["Diff"]-mean_diff) > zfc*std_diff)[0]
        if len(idx) > 0:
            iycscz["passed"] = False
            if required_label not in list(results["jumps"].keys()):
                results["jumps"][required_label] = {}
            results["jumps"][required_label]["zfc"] = zfc
            results["jumps"][required_label]["number"] = len(idx)
            results["jumps"][required_label]["dates"] = var["DateTime"][idx]
            results["jumps"][required_label]["values"] = var["Data"][idx]
            results["jumps"][required_label]["diffs"] = var["Diff"][idx]
            results["jumps"][required_label]["indices"] = idx.copy()
    if len(list(results["jumps"].keys())) > 0:
        msg = "check_step_change_zscore: "
        msg += ",".join(list(results["jumps"].keys())) + " have large jumps"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
        for label in list(results["jumps"].keys()):
            zfc = results["jumps"][label]["zfc"]
            number = results["jumps"][label]["number"]
            msg = "**** " + label + ": " + str(number) + " jumps > " + str(zfc) + "*std(diff)"
            messages["ERROR"].append(msg)
            to_pdf["messages"].append(msg)
            logger.info(msg)
    else:
        msg = "check_step_change_zscore: all required variables passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return iycscz["passed"]
def check_phase_shift(cfg, ds, info):
    info["check_phase_shift"] = init_info()
    iycps = info["check_phase_shift"]
    messages = iycps["messages"]
    results = iycps["results"]
    results = {"Start": [], "End": [], "Lag": [], "Correlation": []}
    to_pdf = iycps["to_pdf"]
    nrecs = int(ds.root["Attributes"]["nc_nrecs"])
    ts = int(ds.root["Attributes"]['time_step'])
    ntsInDay = int(24.0*60.0/float(ts))
    #nDays = nrecs//ntsInDay
    window_days = 15
    window_records = window_days*ntsInDay
    Fsd = pfp_utils.GetVariable(ds, "Fsd", match="wholedays")
    Fsd_toa = pfp_utils.GetVariable(ds, "Fsd_toa", match="wholedays")
    window = {}
    envelope = {}
    si = 0
    ei = window_records
    while ei < nrecs:
        window["Fsd"] = Fsd["Data"][si:ei]
        window["Fsd_toa"] = Fsd_toa["Data"][si:ei]
        window["DateTime"] = Fsd["DateTime"][si:ei]
        window["Hour"] = numpy.ma.array([dt.hour+dt.minute/60 for dt in window["DateTime"]])
        if numpy.ma.count(window["Fsd"]) > window_records/2:
            sdt = window["DateTime"][0]
            edt = window["DateTime"][-1]
            window["Fsd"] = window["Fsd"].reshape(window_days, ntsInDay)
            window["Fsd_toa"] = window["Fsd_toa"].reshape(window_days, ntsInDay)
            window["Hour"] = window["Hour"].reshape(window_days, ntsInDay)
            envelope["Fsd"] = numpy.ma.max(window["Fsd"], axis=0)
            envelope["Fsd_toa"] = numpy.ma.max(window["Fsd_toa"], axis=0)
            envelope["Hour"] = numpy.ma.max(window["Hour"], axis=0)
            lags, corr = get_lagged_correlation(envelope["Fsd_toa"], envelope["Fsd"], 24)
            results["Start"].append(sdt)
            results["End"].append(edt)
            results["Lag"].append(lags[numpy.argmax(corr)])
            results["Correlation"].append(numpy.max(corr))
        si = ei
        ei = ei + window_records
    results["Start"] = numpy.array(results["Start"])
    results["End"] = numpy.array(results["End"])
    results["Lag"] = numpy.array(results["Lag"])
    results["Correlation"] = numpy.array(results["Correlation"])
    idx = numpy.where(abs(results["Lag"]) > 1)[0]
    if len(idx) > 0:
        info["check_phase_shift"]["passed"] = False
        msg = "check_phase_shift: " + str(len(idx)) + " 15-day windows have ABS(lag) > 1"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    else:
        msg = "check_phase_shift: no phase shift in Fsd detected"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return info["check_phase_shift"]["passed"]
def check_diurnal_ipr(cfg, ds, info):
    info["check_diurnal_ipr"] = init_info()
    info["check_diurnal_ipr"]["passed"] = True
    iycdi = info["check_diurnal_ipr"]
    messages = iycdi["messages"]
    results = iycdi["results"]
    to_pdf = iycdi["to_pdf"]
    outlier_labels = []
    ds_labels = list(ds.root["Variables"].keys())
    #nrecs = int(ds.root["Attributes"]["nc_nrecs"])
    ts = int(ds.root["Attributes"]["time_step"])
    ntsInDay = int(24.0*60.0/float(ts))
    ldt = pfp_utils.GetVariable(ds, "DateTime", match="wholedays")
    tdt = ldt["Data"] - datetime.timedelta(minutes=ts)
    months = numpy.array([dt.month for dt in tdt])
    labels = [l for l in cfg["Variables"].keys() if "IPRCheck" in cfg["Variables"][l]]
    for label in labels:
        if label not in ds_labels:
            continue
        results[label] = {}
        messages["outlier_months"] = []
        var = pfp_utils.GetVariable(ds, label, match="wholedays")
        cfgs = ["Variables", label, "IPRCheck"]
        opt = pfp_utils.get_keyvaluefromcf(cfg, cfgs, "upper_percentile", default=90)
        hip = float(opt)
        opt = pfp_utils.get_keyvaluefromcf(cfg, cfgs, "lower_percentile", default=10)
        lop = float(opt)
        opt = pfp_utils.get_keyvaluefromcf(cfg, cfgs, "num_ipr", default=1.5)
        num_ipr = float(opt)
        opt = pfp_utils.get_keyvaluefromcf(cfg, cfgs, "min_ipr", default=1.1)
        min_ipr = float(opt)
        for month in range(1, 13):
            midx = numpy.where(months == month)[0]
            if len(midx) == 0:
                continue
            monthly = {}
            monthly["Data"] = var["Data"][midx]
            monthly["DateTime"] = var["DateTime"][midx]
            monthly["2D"] = monthly["Data"].reshape(-1, ntsInDay)
            nDays = monthly["2D"].shape[0]
            monthly["count"] = numpy.ma.count(monthly["2D"], axis=0)
            monthly["mean"] = numpy.ma.mean(monthly["2D"], axis=0)
            monthly["std"] = numpy.ma.std(monthly["2D"], axis=0)
            monthly["max"] = numpy.ma.max(monthly["2D"], axis=0)
            monthly["min"] = numpy.ma.min(monthly["2D"], axis=0)
            filled = numpy.ma.filled(monthly["2D"], numpy.nan)
            monthly["hip"] = numpy.nanpercentile(filled, hip, axis=0)
            monthly["lop"] = numpy.nanpercentile(filled, lop, axis=0)
            monthly["ipr"] = monthly["hip"] - monthly["lop"]
            monthly["upr"] = monthly["hip"] + num_ipr*monthly["ipr"]
            monthly["lwr"] = monthly["lop"] - num_ipr*monthly["ipr"]
            cond = monthly["upr"] == monthly["lwr"]
            uprc = min_ipr*numpy.ma.max(monthly["max"])
            uprc = monthly["upr"] + uprc
            monthly["uprc"] = numpy.where(cond, uprc, monthly["upr"])
            lwrc = min_ipr*numpy.ma.max(monthly["max"])
            lwrc = monthly["lwr"] - lwrc
            monthly["lwrc"] = numpy.where(cond, lwrc, monthly["lwr"])
            cond1 = monthly["Data"] >= numpy.tile(monthly["uprc"], nDays)
            cond2 = monthly["Data"] <= numpy.tile(monthly["lwrc"], nDays)
            monthly["outside"] = numpy.ma.where(cond1 | cond2)[0]
            cond1 = monthly["Data"] < numpy.tile(monthly["uprc"], nDays)
            cond2 = monthly["Data"] > numpy.tile(monthly["lwrc"], nDays)
            monthly["inside"] = numpy.ma.where(cond1 & cond2)[0]
            results[label][str(month)] = monthly
            if len(monthly["outside"]) > 0:
                msg = f"({month}, {len(monthly["outside"])})"
                messages["outlier_months"].append(msg)
        if len(messages["outlier_months"]) > 0:
            info["check_diurnal_ipr"]["passed"] = False
            msg = f"**** {label} has IPR outliers in {len(messages["outlier_months"])} months"
            messages["ERROR"].append(msg)
            if label not in outlier_labels:
                outlier_labels.append(label)
    if len(outlier_labels) > 0:
        msg = "check_diurnal_ipr: "
        msg += ",".join(outlier_labels) + " have outliers"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
        for msg in messages["ERROR"][:-1]:
            to_pdf["messages"].append(msg)
            logger.error(msg)
    return info["check_diurnal_ipr"]["passed"]

def check_multivariate_comparisons(cfg, ds, info):
    """
    Check comparisons between variables.
    """
    passed = ta_tsonic_cross_check(cfg, ds, info)
    if not passed:
        plot_ta_tsonic_cross_check(cfg, ds, info)
    #ppfd_in_sw_in_cross_check()
    passed = ws_ustar_cross_check(cfg, ds, info)
    if not passed:
        plot_ws_ustar_cross_check(cfg, ds, info)
    return

def get_UTCfromlocaltime(ds):
    tz = ds.root["Attributes"]["time_zone"]
    loc_tz = pytz.timezone(tz)
    ldt = ds.root["Variables"]["DateTime"]["Data"]
    ldt_loc = numpy.array([loc_tz.localize(dt) for dt in ldt])
    ldt_loc_nodst = numpy.array([dt+dt.dst() for dt in ldt_loc])
    ldt_utc = numpy.array([dt.astimezone(pytz.utc) for dt in ldt_loc_nodst])
    return ldt_utc
def get_contiguous_regions(ds, idx, min_recs):
    nrecs = int(ds.root["Attributes"]["nc_nrecs"])
    cidx = numpy.zeros(nrecs)
    cidx[idx] = 1
    for start, stop in pfp_utils.contiguous_regions(cidx):
        duration = stop - start
        if duration < 6:
            cidx[start: stop+1] = 0
    return numpy.where(cidx == 1)[0]
def get_downwelling_shortwave_toa(ds):
    nrecs = int(float(ds.root["Attributes"]["nc_nrecs"]))
    Fsd_toa = pfp_utils.CreateEmptyVariable("Fsd_toa", nrecs)
    solar_altitude = pfp_utils.CreateEmptyVariable("solar_altitude", nrecs)
    lat = float(ds.root["Attributes"]["latitude"])
    lon = float(ds.root["Attributes"]["longitude"])
    ldt_UTC = get_UTCfromlocaltime(ds)
    sa = numpy.array([pysolar.GetAltitude(lat, lon, dt) for dt in ldt_UTC])
    day = numpy.array([pysolar.GetDayOfYear(dt) for dt in ldt_UTC])
    flux_norm = numpy.array([pysolar.GetApparentExtraterrestrialFlux(d) for d in day])
    flux_toa = flux_norm * numpy.sin(numpy.radians(sa))
    flux_toa = numpy.where(sa <= 0, 0, flux_toa)
    solar_altitude["Data"] = numpy.ma.array(sa)
    solar_altitude["Flag"] = numpy.zeros(nrecs, dtype=numpy.int32)
    solar_altitude["Attr"] = {"long_name": "Solar altitude",
                              "units": "degrees", "statistic_type": "average"}
    pfp_utils.CreateVariable(ds, solar_altitude)
    Fsd_toa["Data"] = numpy.ma.array(flux_toa)
    Fsd_toa["Flag"] = numpy.zeros(nrecs, dtype=numpy.int32)
    Fsd_toa["Attr"] = {"long_name": "Downwelling shortwave radiation, top of atmosphere",
                        "units": "W/m^2", "statistic_type": "average"}
    pfp_utils.CreateVariable(ds, Fsd_toa)
    return
def get_day_night_indicator(ds, Fsd_label='Fsd_toa'):
    nrecs = int(ds.root["Attributes"]["nc_nrecs"])
    Fsd = pfp_utils.GetVariable(ds, Fsd_label)
    day_night = pfp_utils.CreateEmptyVariable("day_night", nrecs)
    zeros = numpy.zeros(nrecs)
    ones = numpy.ones(nrecs)
    day_night["Data"] = numpy.ma.where(Fsd["Data"] > 10, ones, zeros)
    day_night["Flag"] = numpy.zeros(nrecs, dtype=numpy.int32)
    day_night["Attr"] = {"long_name": "Day/night indicator, 1=day, 0=night",
                         "comment": "Derived using "+Fsd_label,
                         "units": 1, "statistic_type": "average"}
    pfp_utils.CreateVariable(ds, day_night)
    return
def get_lagged_correlation(x_in, y_in, maxlags):
    """
    Calculate the lagged cross-correlation between 2 1D arrays.
    Taken from the matplotlib.pyplot.xcorr source code.
    PRI added handling of masked arrays.
    """
    lags = numpy.arange(-maxlags, maxlags+1)
    mask = numpy.ma.mask_or(x_in.mask, y_in.mask, copy=True, shrink=False)
    x = numpy.ma.array(x_in, mask=mask, copy=True)
    y = numpy.ma.array(y_in, mask=mask, copy=True)
    x = numpy.ma.compressed(x)
    y = numpy.ma.compressed(y)
    corr = numpy.correlate(x, y, mode=2)
    corr/= numpy.sqrt(numpy.dot(x, x) * numpy.dot(y, y))
    if maxlags is None: maxlags = len(x) - 1
    if maxlags >= len(x) or maxlags < 1:
        raise ValueError('pfp_ts.get_laggedcorrelation: maxlags must be None or strictly positive < %d'%len(x))
    corr = corr[len(x) - 1 - maxlags:len(x) + maxlags]
    return lags, corr

def init_pdf_report(site_name):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page('P')
    pdf.set_font("Arial", size=16, style='B')
    txt = "Data Analysis Report: " + site_name
    pdf.cell(200, 10, txt=txt, ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    return pdf

def pad_masked_array(var, pad_width):
    """
    Pads a 1D masked array using 'reflect' logic for both data and mask.
    """
    n = len(var["Data"])
    new_shape = n + 2 * pad_width
    # Create empty masked array with the same dtype
    padded = numpy.ma.masked_all(new_shape, dtype=var["Data"].dtype)
    # Place original data in the center
    padded[pad_width : pad_width + n] = var["Data"]
    # Manually reflect the edges for both data and mask
    # Left edge
    padded[:pad_width] = var["Data"][1 : pad_width + 1][::-1]
    # Right edge
    padded[pad_width + n:] = var["Data"][n - pad_width - 1 : n - 1][::-1]
    return padded

def plot_diurnal_ipr(cfg, ds, info, year):
    iycdi = info["check_diurnal_ipr"]
    to_pdf = iycdi["to_pdf"]
    site_name = ds.root["Attributes"]["site_name"]
    site_name = site_name.replace(" ", "")
    ldt = pfp_utils.GetVariable(ds, "DateTime")
    sdt = ldt["DateTime"][0]
    edt = ldt["DateTime"][-1]
    ncols = 4
    nrows = 4
    nfig = 1
    naxes = 0
    hr = numpy.arange(0, 24, 0.5)
    ipr_labels = list(iycdi["results"].keys())
    for ipr_label in ipr_labels:
        months = list(iycdi["results"][ipr_label].keys())
        for month in months:
            if naxes < ncols*nrows:
                ncol = numpy.mod(naxes, ncols)
                nrow = naxes//ncols
                monthly = iycdi["results"][ipr_label][month]
                data = monthly["Data"]
                ldt = monthly["DateTime"]
                hm = numpy.array([dt.hour+dt.minute/60 for dt in ldt])
                uprc = monthly["uprc"]
                lwrc = monthly["lwrc"]
                inside = monthly["inside"]
                outside = monthly["outside"]
                percent_outside = 100*len(monthly["outside"])/len(monthly["Data"])
                if percent_outside <= 0.5:
                    continue
                if not plt.fignum_exists(nfig):
                    fig, axs = plt.subplots(num=nfig, nrows=nrows, ncols=ncols, sharex=True, figsize=(11,8))
                    title_str = site_name + ": Diurnal IPR checks; " + sdt.strftime("%Y-%m-%d")
                    title_str += " to " + edt.strftime("%Y-%m-%d")
                    fig.suptitle(title_str)
                axs[nrow, ncol].plot(hm[inside], data[inside], 'b.', alpha=0.25)
                axs[nrow, ncol].plot(hm[outside], data[outside], 'ro')
                axs[nrow, ncol].plot(hr, uprc, 'r--')
                axs[nrow, ncol].plot(hr, lwrc, 'r--')
                axs[nrow, ncol].set_xticks([0, 6, 12, 18, 24])
                if nrow == nrows-1:
                    axs[nrow, ncol].set_xlabel("Hour")
                axs[nrow, ncol].set_ylabel(ipr_label)
                text = f"{int(year):04d}{int(month):02d}"
                axs[nrow, ncol].text(0.75, 0.9, text, fontsize=8, transform=axs[nrow, ncol].transAxes)
                text = f"{len(data)},{len(inside)},{len(outside)}"
                axs[nrow, ncol].text(0.05, 0.9, text, fontsize=8, transform=axs[nrow, ncol].transAxes)
                naxes += 1
            else:
                naxes = 0
                fig.tight_layout()
                figure_name = site_name + "_outliers_IPR_" + str(year) + "_" + str(nfig) + ".png"
                figure_uri = os.path.join("plots", figure_name)
                fig.savefig(figure_uri, format='png')
                to_pdf["images"].append(figure_uri)
                plt.show()
                nfig += 1
                fig, axs = plt.subplots(num=nfig, nrows=nrows, ncols=ncols, sharex=True, figsize=(11,8))
                title_str = site_name + ": Diurnal IPR checks; " + sdt.strftime("%Y-%m-%d")
                title_str += " to " + edt.strftime("%Y-%m-%d")
                fig.suptitle(title_str)
    if plt.fignum_exists(nfig):
        for ncol in range(ncols):
            for nrow in range(nrows):
                if not axs[nrow, ncol].has_data():
                    axs[nrow, ncol].remove()
        if len(fig.get_axes()) == 0:
            plt.close(nfig)
        else:
            fig.tight_layout()
            figure_name = site_name + "_outliers_IPR_" + str(year) + "_" + str(nfig) + ".png"
            figure_uri = os.path.join("plots", figure_name)
            fig.savefig(figure_uri, format='png')
            to_pdf["images"].append(figure_uri)
            plt.show()
    return
def plot_ta_tsonic_cross_check(cfg, ds, info):
    itatscc = info["ta_tsonic_cross_check"]
    results = itatscc["results"]
    to_pdf = itatscc["to_pdf"]
    site_name = ds.root["Attributes"]["site_name"]
    Ta = results["Ta"]
    Ta_SONIC_Av = results["Ta_SONIC_Av"]
    figure_name = site_name + "_ta_tsonic_cross_check_" + str(year) + "_.png"
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    window_title = site_name + ": Ta vs Ta_SONIC_Av cross check"
    fig.canvas.manager.set_window_title(window_title)
    axs.plot(Ta["Data"], Ta_SONIC_Av["Data"], 'b.', alpha=0.25)
    axs.set_xlabel(Ta["Label"]+"("+Ta["Attr"]["units"]+")")
    axs.set_ylabel(Ta_SONIC_Av["Label"]+"("+Ta_SONIC_Av["Attr"]["units"]+")")
    fig.tight_layout()
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    plt.show()
    return figure_name
def plot_ws_ustar_cross_check(cfg, ds, info):
    iwsuscc = info["ws_ustar_cross_check"]
    results = iwsuscc["results"]
    to_pdf = iwsuscc["to_pdf"]
    site_name = ds.root["Attributes"]["site_name"]
    Ws = results["Ws"]
    ustar = results["ustar"]
    figure_name = site_name + "_ws_ustar_cross_check_" + str(year) + "_.png"
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    window_title = site_name + ": Ws vs ustar cross check"
    fig.canvas.manager.set_window_title(window_title)
    axs.plot(Ws["Data"], ustar["Data"], 'b.', alpha=0.25)
    axs.set_xlabel(Ws["Label"]+"("+Ws["Attr"]["units"]+")")
    axs.set_ylabel(ustar["Label"]+"("+ustar["Attr"]["units"]+")")
    fig.tight_layout()
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    plt.show()
    return figure_name
def plot_outliers_ranges(cfg, ds, info):
    iycor = info["check_outliers_ranges"]
    #messages = iycor["messages"]
    results = iycor["results"]
    to_pdf = iycor["to_pdf"]
    outlier_labels = list(results["outliers"].keys())
    nrows = len(outlier_labels)
    site_name = ds.root["Attributes"]["site_name"]
    figure_name = site_name + "_outliers_ranges_" + str(year) + "_.png"
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    axs = fig.get_axes()
    window_title = site_name + ": outliers"
    fig.canvas.manager.set_window_title(window_title)
    for n, outlier_label in enumerate(outlier_labels):
        upper = float(cfg["Variables"][outlier_label]["RangeCheck"]["upper"])
        lower = float(cfg["Variables"][outlier_label]["RangeCheck"]["lower"])
        var = pfp_utils.GetVariable(ds, outlier_label)
        idx_upper = numpy.ma.where(var["Data"] > upper)[0]
        idx_lower = numpy.ma.where(var["Data"] < lower)[0]
        sdt = var["DateTime"][0]
        edt = var["DateTime"][-1]
        if n == 0:
            title_str = site_name + ": Outliers (ranges); " + sdt.strftime("%Y-%m-%d")
            title_str += " to " + edt.strftime("%Y-%m-%d")
            axs[n].set_title(title_str)
        legend_label = str(len(idx_upper)) + " >" + str(upper) + "; "
        legend_label += str(len(idx_lower)) + " <" + str(lower)
        axs[n].plot(var["DateTime"], var["Data"], 'b.',
                    label=legend_label, alpha=0.25)
        if len(idx_upper) > 0:
            axs[n].plot(var["DateTime"][idx_upper], var["Data"][idx_upper], 'ro')
            axs[n].plot([sdt, edt], [upper, upper], 'r--')
        if len(idx_lower) > 0:
            axs[n].plot(var["DateTime"][idx_lower], var["Data"][idx_lower], 'ro')
            axs[n].plot([sdt, edt], [lower, lower], 'r--')
        axs[n].set_xlim([sdt, edt])
        axs[n].set_ylabel(outlier_label)
        axs[n].legend()
    fig.tight_layout()
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    #fig.canvas.flush_events()
    plt.show()
    return figure_name
def plot_outliers_gmad(cfg, ds, info, year):
    """ Plot noise spikes identified by the Gemini MAD filter function."""
    iycog = info["check_outliers_gmad"]
    results = iycog["results"]
    to_pdf = iycog["to_pdf"]
    outlier_labels = list(results["outliers"].keys())
    nrows = len(outlier_labels)
    site_name = ds.root["Attributes"]["site_name"]
    ldt = pfp_utils.GetVariable(ds, "DateTime")
    sdt = ldt["DateTime"][0]
    edt = ldt["DateTime"][-1]
    figure_name = site_name + "_outliers_gmad_" + str(year) + "_.png"
    fig, _ = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    axs = fig.get_axes()
    window_title = site_name + ": Outliers (GMAD)"
    fig.canvas.manager.set_window_title(window_title)
    title_str = site_name + ": Outliers (GMAD); " + sdt.strftime("%Y-%m-%d")
    title_str += " to " + edt.strftime("%Y-%m-%d")
    fig.suptitle(title_str)
    for n, outlier_label in enumerate(outlier_labels):
        var = pfp_utils.GetVariable(ds, outlier_label)
        axs[n].plot(var["DateTime"], var["Data"], 'b.', alpha=0.25)
        idx = results["outliers"][outlier_label]["indices"]
        label = "n=" + str(len(idx))
        axs[n].plot(var["DateTime"][idx], var["Data"][idx], 'ro', label=label)
        axs[n].set_xlim([sdt, edt])
        axs[n].set_ylabel(outlier_label)
        axs[n].legend()
    fig.tight_layout()
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    #fig.canvas.flush_events()
    plt.show()
    return
def plot_outliers_imad(cfg, ds, info, year):
    iycoi = info["check_outliers_imad"]
    results = iycoi["results"]
    to_pdf = iycoi["to_pdf"]
    outlier_labels = list(results["outliers"].keys())
    nrows = len(outlier_labels)
    site_name = ds.root["Attributes"]["site_name"]
    figure_name = site_name + "_outliers_imad_" + str(year) + "_.png"
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    window_title = site_name + ": outliers (IMAD)"
    fig.canvas.manager.set_window_title(window_title)
    for n, outlier_label in enumerate(outlier_labels):
        var = pfp_utils.GetVariable(ds, outlier_label)
        sdt = var["DateTime"][0]
        edt = var["DateTime"][-1]
        if n == 0:
            title_str = site_name + ": " + sdt.strftime("%Y-%m-%d") + " to "
            title_str += edt.strftime("%Y-%m-%d")
            axs[n].set_title(title_str)
        axs[n].plot(var["DateTime"], var["Data"], 'b.', label=outlier_label, alpha=0.25)
        idx = results["outliers"][outlier_label]["indices"]
        axs[n].plot(var["DateTime"][idx], var["Data"][idx], 'ro')
        axs[n].legend()
    fig.tight_layout()
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    #fig.canvas.flush_events()
    plt.show()
    return
def plot_radiation(cfg, ds, info, year):
    iycs = info["check_radiation"]
    #messages = iycs["messages"]
    results = iycs["results"]
    to_pdf = iycs["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    ldt = pfp_utils.GetVariable(ds, "DateTime")
    sdt = ldt["DateTime"][0]
    edt = ldt["DateTime"][-1]
    if "Fsd" not in ds_labels:
        msg = " Fsd not found in data structure"
        logger.error(msg)
        return ""
    site_name = ds.root["Attributes"]["site_name"]
    figure_name = site_name+"_radiation_" + str(year) + ".png"
    data = {}
    for label in ["Fsd", "Fsu", "Fld", "Flu"]:
        if label in ds_labels:
            data[label] = pfp_utils.GetVariable(ds, label)
    nrows = len(data.keys())
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(11,8))
    window_title = site_name + ": Shortwave Radiation TOA"
    fig.canvas.manager.set_window_title(window_title)
    title_str = site_name + ": Radiation checks; " + sdt.strftime("%Y-%m-%d")
    title_str += " to " + edt.strftime("%Y-%m-%d")
    fig.suptitle(title_str)
    for n, label in enumerate(list(data.keys())):
        ncol = 1
        x = data[label]["DateTime"]
        y = data[label]["Data"]
        axs[n].plot(x, y, 'b.', alpha=0.25)
        axs[n].set_ylabel(label)
        axs[n].set_xlim([sdt, edt])
        #axs[n].tick_params(labelbottom=True)
        #axs[n].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        if label == "Fsd":
            if "dgtidx" in results:
                dgtidx = results["dgtidx"]
                x = data[label]["DateTime"][dgtidx]
                y = data[label]["Data"][dgtidx]
                axs[n].plot(x, y, 'r.', label='Fsd>1.1*TOA')
                ncol += 1
            if "nltidx" in results:
                nltidx = results["nltidx"]
                x = data[label]["DateTime"][nltidx]
                y = data[label]["Data"][nltidx]
                axs[n].plot(x, y, 'g.', label='Fsd[night]<-10')
                ncol += 1
            if "ngtidx" in results:
                ngtidx = results["ngtidx"]
                x = data[label]["DateTime"][ngtidx]
                y = data[label]["Data"][ngtidx]
                axs[n].plot(x, y, 'y.', label='Fsd[night]>10')
                ncol += 1
            axs[n].legend(ncol=ncol)
        if ((label == "Flu") or (label == "Fld")):
            if "fldgtflu" in results:
                fldgtflu = results["fldgtflu"]
                x = data[label]["DateTime"][fldgtflu]
                y = data[label]["Data"][fldgtflu]
                axs[n].plot(x, y, 'r.', label='Fld>Flu')
                ncol += 1
            axs[n].legend(ncol=ncol)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    plt.show()
    return figure_name
def plot_step_change_jump(cfg, ds, info):
    """ Plot step changes identified by jump in values exceeding threshold."""
    iycscj = info["check_step_change_jump"]
    results = iycscj["results"]
    to_pdf = iycscj["to_pdf"]
    jump_labels = list(results["jumps"].keys())
    nrows = len(jump_labels)
    site_name = ds.root["Attributes"]["site_name"]
    ldt = pfp_utils.GetVariable(ds, "DateTime")
    sdt = ldt["DateTime"][0]
    edt = ldt["DateTime"][-1]
    figure_name = site_name + "_jumps_" + str(year) + "_.png"
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    window_title = site_name + ": jumps"
    fig.canvas.manager.set_window_title(window_title)
    title_str = site_name + ": Step change (jump); " + sdt.strftime("%Y-%m-%d")
    title_str += " to " + edt.strftime("%Y-%m-%d")
    fig.suptitle(title_str)
    for n, jump_label in enumerate(jump_labels):
        var = pfp_utils.GetVariable(ds, jump_label)
        axs[n].plot(var["DateTime"], var["Data"], 'b.', alpha=0.25)
        idx = results["jumps"][jump_label]["indices"]
        label = "n=" + str(len(idx))
        axs[n].plot(var["DateTime"][idx], var["Data"][idx], 'ro', label=label)
        axs[n].set_ylabel(jump_label)
        axs[n].set_xlim([sdt, edt])
        axs[n].legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    #fig.canvas.flush_events()
    plt.show()
    return
def plot_step_change_zscore(cfg, ds, info):
    """ Plot step changes identified by jump in values exceeding threshold."""
    iycscz = info["check_step_change_zscore"]
    results = iycscz["results"]
    to_pdf = iycscz["to_pdf"]
    jump_labels = list(results["jumps"].keys())
    nrows = len(jump_labels)
    site_name = ds.root["Attributes"]["site_name"]
    ldt = pfp_utils.GetVariable(ds, "DateTime")
    sdt = ldt["DateTime"][0]
    edt = ldt["DateTime"][-1]
    figure_name = site_name + "_zscore_" + str(year) + "_.png"
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                            figsize=(11, 8), tight_layout=True)
    window_title = site_name + ": Z-score"
    fig.canvas.manager.set_window_title(window_title)
    title_str = site_name + ": Step change (Z score); " + sdt.strftime("%Y-%m-%d")
    title_str += " to " + edt.strftime("%Y-%m-%d")
    fig.suptitle(title_str)
    for n, jump_label in enumerate(jump_labels):
        var = pfp_utils.GetVariable(ds, jump_label)
        axs[n].plot(var["DateTime"], var["Data"], 'b.', alpha=0.25)
        idx = results["jumps"][jump_label]["indices"]
        label = "n=" + str(len(idx))
        axs[n].plot(var["DateTime"][idx], var["Data"][idx], 'ro', label=label)
        axs[n].legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    figure_uri = os.path.join("plots", figure_name)
    fig.savefig(figure_uri, format='png')
    to_pdf["images"].append(figure_uri)
    #fig.canvas.flush_events()
    plt.show()
    return
def SubsetDataStructure(ds_all, start=0, end=-1, subset_labels="all"):
    # sanity checks
    if isinstance(subset_labels, str):
        if subset_labels.lower() == "all":
            labels = list(ds_all.root["Variables"].keys())
    elif isinstance(subset_labels, list):
        labels = list(subset_labels)
    else:
        msg = " Unrecognised option for subset_labels"
        print(msg)
        return copy.deepcopy(ds_all)
    ds = pfp_io.DataStructure()
    ds.root["Attributes"] = copy.deepcopy(ds_all.root["Attributes"])
    ds_labels = list(ds_all.root["Variables"].keys())
    for label in labels:
        if label in ds_labels:
            var = pfp_utils.GetVariable(ds_all, label, start=start, end=end)
            pfp_utils.CreateVariable(ds, var)
    nrecs = len(ds.root["Variables"]["time"]["Data"])
    ds.root["Attributes"]["nc_nrecs"] = nrecs
    ldt = ds.root["Variables"]["DateTime"]["Data"]
    ds.root["Attributes"]["time_coverage_start"] = str(ldt[0])
    ds.root["Attributes"]["time_coverage_end"] = str(ldt[-1])
    return ds
def ta_tsonic_cross_check(cfg, ds, info):
    info["ta_tsonic_cross_check"] = init_info()
    itatscc = info["ta_tsonic_cross_check"]
    itatscc["passed"] = True
    messages = itatscc["messages"]
    results = itatscc["results"]
    to_pdf = itatscc["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    # check to see if required variables are in the data structure
    if "Ta" not in ds_labels:
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Ta not in data structure"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if "Ta_SONIC_Av" not in ds_labels:
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Ta_SONIC_Av not in data structure"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if not itatscc["passed"]:
        return itatscc["passed"]
    # get the variables
    Ta = pfp_utils.GetVariable(ds, "Ta")
    results["Ta"] = Ta
    Ta_SONIC_Av = pfp_utils.GetVariable(ds, "Ta_SONIC_Av")
    results["Ta_SONIC_Av"] = Ta_SONIC_Av
    # correlation, slope and intercept check
    corr_coef = numpy.ma.corrcoef(Ta["Data"], Ta_SONIC_Av["Data"])
    results["corr_coef"] = corr_coef
    if corr_coef[0, 1] < 0.7:
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Pearson correlation coefficient is less than 0.7"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    # slope and intercept
    slope, intercept = numpy.ma.polyfit(Ta["Data"], Ta_SONIC_Av["Data"], 1)
    results["slope"] = slope
    results["intercept"] = intercept
    if ((slope < 0.8) or (slope > 1.2)):
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Slope of Ta vs Ta_SONIC_Av outside range 0.8 to 1.2"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if abs(intercept) > 3:
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Intercept of Ta vs Ta_SONIC_Av outside range -3 to 3 degC"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    # mean and standard deviation difference
    bias = abs(numpy.ma.mean(Ta_SONIC_Av["Data"]) - numpy.ma.mean(Ta["Data"]))
    results["bias"] = bias
    if bias > 10:
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Difference between means is greater than 10 degC"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    std_diff = abs(numpy.ma.std(Ta_SONIC_Av["Data"]) - numpy.ma.std(Ta["Data"]))
    results["std_diff"] = std_diff
    if std_diff > 2:
        itatscc["passed"] = False
        msg = "ta_tsonic_cross_check: Difference between standard deviations is greater than 2 degC"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if itatscc["passed"]:
        msg = "ta_tsonic_cross_check: all tests passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return itatscc["passed"]
def ws_ustar_cross_check(cfg, ds, info):
    info["ws_ustar_cross_check"] = init_info()
    iwsuscc = info["ws_ustar_cross_check"]
    iwsuscc["passed"] = True
    messages = iwsuscc["messages"]
    results = iwsuscc["results"]
    to_pdf = iwsuscc["to_pdf"]
    ds_labels = list(ds.root["Variables"].keys())
    # check to see if required variables are in the data structure
    if "Ws" not in ds_labels:
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: Ws not in data structure"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if "ustar" not in ds_labels:
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: ustar not in data structure"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if not iwsuscc["passed"]:
        return iwsuscc["passed"]
    # get the variables
    Ws = pfp_utils.GetVariable(ds, "Ws")
    results["Ws"] = Ws
    ustar = pfp_utils.GetVariable(ds, "ustar")
    results["ustar"] = ustar
    # correlation, slope and intercept check
    corr_coef = numpy.ma.corrcoef(Ws["Data"], ustar["Data"])
    results["corr_coef"] = corr_coef
    if corr_coef[0, 1] < 0.7:
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: Pearson correlation coefficient is less than 0.7"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    # slope and intercept
    slope, intercept = numpy.ma.polyfit(Ws["Data"], ustar["Data"], 1)
    results["slope"] = slope
    results["intercept"] = intercept
    if ((slope < 0.05) or (slope > 0.2)):
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: Slope of Ws vs ustar outside range 0.05 to 0.2"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if abs(intercept) > 1:
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: Intercept of Ws vs ustar outside range -1 to 1 degC"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    # mean and standard deviation difference
    bias = abs(numpy.ma.mean(ustar["Data"]) - numpy.ma.mean(Ws["Data"]))
    results["bias"] = bias
    if bias > 10:
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: Difference between means is greater than 10 m/s"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    std_diff = abs(numpy.ma.std(ustar["Data"]) - numpy.ma.std(Ws["Data"]))
    results["std_diff"] = std_diff
    if std_diff > 2:
        iwsuscc["passed"] = False
        msg = "ws_ustar_cross_check: Difference between standard deviations is greater than 2 m/s"
        messages["ERROR"].append(msg)
        to_pdf["messages"].append(msg)
        logger.error(msg)
    if iwsuscc["passed"]:
        msg = "ws_ustar_cross_check: all tests passed"
        messages["INFO"].append(msg)
        to_pdf["messages"].append(msg)
        logger.info(msg)
    return iwsuscc["passed"]

if (__name__ == '__main__'):
    # initialise a dictionary to hold the QC messages
    messages = {}
    # get the auto_qc control file contents
    cfg_uri = "auto_qc.txt"
    cfg = ConfigObj(cfg_uri, indent_type="    ", list_values=False)

    #nc_uri = os.path.join("data", "AliceSpringsMulga1_L3.nc")
    nc_uri = os.path.join("data", "Boyagin_L3.nc")
    ds_all = pfp_io.NetCDFRead(nc_uri)
    ts = int(ds_all.root["Attributes"]["time_step"])
    tdts = datetime.timedelta(minutes=ts)
    site_name = ds_all.root["Attributes"]["site_name"]
    site_name = site_name.replace(" ", "")
    ldt = pfp_utils.GetVariable(ds_all, "DateTime")
    first_year = (ldt["Data"][0]-tdts).year
    last_year = (ldt["Data"][-1]-tdts).year
    years = range(first_year, last_year+1)

    info = {}
    years = [2024]
    for year in years:
        info[str(year)] = {}
        msg = "QC checks for year " + str(year)
        logger.info(msg)
        start = datetime.datetime(year, 1, 1, 0, 30, 0)
        end = datetime.datetime(year+1, 1, 1, 0, 0, 0)
        ds = SubsetDataStructure(ds_all, start=start, end=end, subset_labels="all")
        # check for time stamp gaps or duplicates
        passed = check_timestep(ds, info[str(year)])
        # check for required variables
        passed = check_required_variables(cfg, ds, info[str(year)])
        # check required variables have at least 75% good data
        passed = check_percent_good(cfg, ds, info[str(year)])
        # check downwelling shortwave radiation against top-of-atmosphere values
        get_downwelling_shortwave_toa(ds)
        get_day_night_indicator(ds)
        passed = check_radiation(cfg, ds, info[str(year)])
        if not passed:
            plot_radiation(cfg, ds, info[str(year)], year)
        # check for phase shift between downwelling shortwave radiation and TOA radiation
        passed = check_phase_shift(cfg, ds, info[str(year)])
        # check for outlier points using plausible ranges
        passed = check_outliers_ranges(cfg, ds, info[str(year)])
        if not passed:
            plot_outliers_ranges(cfg, ds, info[str(year)])
        # check for outliers using diurnal mean +/- 1.5*IPR(90,10)
        passed = check_diurnal_ipr(cfg, ds, info[str(year)])
        if not passed:
            plot_diurnal_ipr(cfg, ds, info[str(year)], year)
        # check for outliers using GMAD (Gemini MAD)
        passed = check_outliers_gmad(cfg, ds, info[str(year)])
        if not passed:
            plot_outliers_gmad(cfg, ds, info[str(year)], year)
        # check for outliers using IMAD (ICOS MAD)
        # too many false positives
        #passed =  check_outliers_imad(cfg, ds, info[str(year)])
        #if not passed:
            #plot_outliers_imad(cfg, ds, info[str(year)], year)
        # check for step changes
        # use absolute ranges read from control file
        passed = check_step_change_jump(cfg, ds, info[str(year)])
        if not passed:
            plot_step_change_jump(cfg, ds, info[str(year)])
        # use Z-score read from control file on differences
        passed = check_step_change_zscore(cfg, ds, info[str(year)])
        if not passed:
            plot_step_change_zscore(cfg, ds, info[str(year)])
        # check for constant values
        # check for multivariate comparisons
        passed = check_multivariate_comparisons(cfg, ds, info[str(year)])
        #if not passed:
            #plot_multivariate_comparisons(cfg, ds, info[str(year)])
        # check SSIM using AWS and ACCESS
        #passed = check_SSIM()
        #if not passed:
            #plot_SSIM()
        # check correlation, slope, intercept and RMSE using AWS and ACCESS
        #passed = check_compare_with_access()
        #if not passed:
            #plot_compare_with_access()
        # write messages to report PDF
        pdf_name = os.path.join("reports", site_name + "_" + str(year) + ".pdf")
        msg = "Writing results to " + pdf_name
        logger.info(msg)
        pdf = init_pdf_report(site_name)
        checks = info[str(year)].keys()
        for check in checks:
            to_pdf = info[str(year)][check]["to_pdf"]["messages"]
            for msg in to_pdf:
                pdf.cell(0, 10, txt=msg, align='J')
                pdf.ln(10)
        for check in checks:
            to_pdf = info[str(year)][check]["to_pdf"]["images"]
            for img in to_pdf:
                pdf.add_page('L')
                pdf.image(img, x=30, y=pdf.get_y(), w=200)
        pdf.output(pdf_name)

        msg =  "Finished " + str(year)
        logger.info(msg)