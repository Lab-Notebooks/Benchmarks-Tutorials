import os
import sys
import json
import itertools
import numpy
import boxkit
import boxkit.resources.flash as flash_box

SIM_LENGTH_SCALE = 1e-3
SIM_TIME_SCALE = 10e-3
SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../simulation/PoolBoiling")
SIM_BASENAME = "INS_Pool_Boiling_hdf5_plt_cnt_"

def read_datasets(dataset_dir, file_tags):
    """
    Read datasets from file tags
    """
    datasets = [boxkit.read_dataset(os.path.join(dataset_dir, SIM_BASENAME + str(tag).zfill(4)),
                                    source="flash") for tag in file_tags]
    return datasets

def process_dataset(dataset):
    """
    Get heat flux profile
    """
    hflux = numpy.array([])
    xloc = numpy.array([])
    iliq = numpy.array([])

    yloc = 0.0

    data_slice = boxkit.create_slice(dataset, ymin=yloc, ymax=yloc)

    for block in data_slice.blocklist:
        yindex = (numpy.abs(block.yrange("center") - yloc)).argmin()
        zindex = 0
        xloc = numpy.append(xloc, block.xrange("center"))
        hflux = numpy.append(hflux,(block["dfun"][zindex,yindex,:]<0)*(1-block["temp"][zindex,yindex,:])/(0.5*block.dy))
        #hflux = numpy.append(hflux,(1-block["temp"][zindex,yindex,:])/(0.5*block.dy))
        iliq = numpy.append(iliq,block["dfun"][zindex,yindex,:]<0)

    mean_hflux = numpy.mean(hflux[:])/numpy.mean(iliq[:])
    #mean_hflux = numpy.mean(hflux[:])

    merged_dataset = boxkit.mergeblocks(dataset, ["dfun"])
    merged_dataset.fill_guard_cells()

    shapelist = boxkit.regionprops(merged_dataset, "dfun")
    if shapelist:
        diameter = 2*numpy.sqrt(2*shapelist[0]["area"]/numpy.pi)
    else:
        diameter = 0.

    return numpy.array([float(dataset.time), float(mean_hflux), float(diameter)])

def ref_comparison_dict():
    """
    Comparison dict for different outflow buffers
    """
    dataset_dir = {}
    dataset_dir["ref-24"] = f"{SIM_PATH}/SingleBubble/reference/refLong-24/jobnode.archive/2023-11-26"
    dataset_dir["ref-12"] = f"{SIM_PATH}/SingleBubble/reference/refLong/jobnode.archive/2023-11-26"
    dataset_dir["ref-6"] = f"{SIM_PATH}/SingleBubble/reference/refShort/jobnode.archive/2023-11-26"

    file_tags = {}
    file_tags["ref-24"] = [*range(100)]
    file_tags["ref-12"] = [*range(77)]
    file_tags["ref-6"] = [*range(100)]

    return dataset_dir, file_tags

def lb_comparison_dict():
    """
    Comparison dict for different outflow buffers
    """
    dataset_dir = {}
    #dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/reference/refLong-24/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/outflow/refLong-24/jobnode.archive/2023-11-28"
    dataset_dir["lb-0.5"] = f"{SIM_PATH}/SingleBubble/outflow/lb_0.5/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["lb-1.0"] = f"{SIM_PATH}/SingleBubble/outflow/lb_1.0/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["lb-1.5"] = f"{SIM_PATH}/SingleBubble/outflow/lb_1.5/jobnode.archive/2023-11-26-propSmear"

    file_tags = {}
    #file_tags["reference"] = [*range(100)]
    file_tags["reference"] = [*range(394)]
    file_tags["lb-0.5"] = [*range(450)]
    file_tags["lb-1.0"] = [*range(444)]
    file_tags["lb-1.5"] = [*range(442)]

    return dataset_dir, file_tags

def lb_comparison_skip_dict():
    """
    Comparison dict for different outflow buffers
    """
    dataset_dir = {}
    #dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/reference/refLong-24/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/outflow/refLong-24/jobnode.archive/2023-11-28"
    dataset_dir["lb-0.5"] = f"{SIM_PATH}/SingleBubble/outflow/lb_0.5/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["lb-1.0"] = f"{SIM_PATH}/SingleBubble/outflow/lb_1.0/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["lb-1.5"] = f"{SIM_PATH}/SingleBubble/outflow/lb_1.5/jobnode.archive/2023-11-26-propSmear"

    file_tags = {}
    #file_tags["reference"] = [*range(100)]
    file_tags["reference"] = [*range(0,155,3)]
    file_tags["lb-0.5"] = [*range(0,450,3)]
    file_tags["lb-1.0"] = [*range(0,444,3)]
    file_tags["lb-1.5"] = [*range(0,442,3)]

    return dataset_dir, file_tags

def lb_noadv_contour_dict():
    """
    Comparison dict for bubble contours
    """
    lb = "lb_1.0"

    dataset_dir = {}
    #dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/outflow/refLong-24/jobnode.archive/2023-11-28"
    dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/reference/refLong-24/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["withAdv"] = f"{SIM_PATH}/SingleBubble/outflow/{lb}/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["withoutAdv"] = f"{SIM_PATH}/SingleBubble/outflow/{lb}/jobnode.archive/2023-11-26-noAdvection"


    tag_list = [0, 10, 20, 30, 40, 50]
    #tag_list = [90, 100, 110, 120, 130, 140]

    file_tags = {}
    file_tags["reference"] = tag_list
    file_tags["withAdv"] = tag_list
    file_tags["withoutAdv"] = tag_list

    return dataset_dir, file_tags


def lb_noadv_comparison_dict():
    """
    Comparison dict for different outflow buffers
    """
    dataset_dir = {}
    #dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/reference/refLong-24/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/outflow/refLong-24/jobnode.archive/2023-11-28"
    dataset_dir["lb-0.5"] = f"{SIM_PATH}/SingleBubble/outflow/lb_0.5/jobnode.archive/2023-11-26-noAdvection"
    dataset_dir["lb-1.0"] = f"{SIM_PATH}/SingleBubble/outflow/lb_1.0/jobnode.archive/2023-11-26-noAdvection"
    dataset_dir["lb-1.5"] = f"{SIM_PATH}/SingleBubble/outflow/lb_1.5/jobnode.archive/2023-11-26-noAdvection"

    file_tags = {}
    file_tags["reference"] = [*range(235)]
    file_tags["lb-0.5"] = [*range(140)]
    file_tags["lb-1.0"] = [*range(220)]
    file_tags["lb-1.5"] = [*range(140)]

    return dataset_dir, file_tags

def lb_noadv_comparison2_dict():
    """
    Comparison dict for different outflow buffers
    """
    lb = "lb_1.0"

    dataset_dir = {}
    #dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/reference/refLong-24/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["reference"] = f"{SIM_PATH}/SingleBubble/outflow/refLong-24/jobnode.archive/2023-11-28"
    dataset_dir["withAdv"] = f"{SIM_PATH}/SingleBubble/outflow/{lb}/jobnode.archive/2023-11-26-propSmear"
    dataset_dir["withoutAdv"] = f"{SIM_PATH}/SingleBubble/outflow/{lb}/jobnode.archive/2023-11-26-noAdvection"

    file_tags = {}
    file_tags["reference"] = [*range(235)]
    file_tags["withAdv"] = [*range(220)]
    file_tags["withoutAdv"] = [*range(220)]

    return dataset_dir, file_tags



if __name__ == "__main__":
    """
    Main
    """
    pass
