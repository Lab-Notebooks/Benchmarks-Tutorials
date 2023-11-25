import os
import numpy
import boxkit
import boxkit.resources.flash as flash_box

SIM_YMIN = -1
SIM_LENGTH_SCALE = 0.5
SIM_TIME_SCALE = 0.71
SIM_SCALE = numpy.array([SIM_TIME_SCALE, SIM_LENGTH_SCALE**2, 1, SIM_LENGTH_SCALE, SIM_LENGTH_SCALE/SIM_TIME_SCALE])
SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../simulation")
SIM_BASENAME = "INS_Rising_Bubble_hdf5_plt_cnt_"

def read_datasets(dataset_dir, file_tags):
    """
    Read datasets from file tags
    """
    datasets = [boxkit.read_dataset(os.path.join(dataset_dir, SIM_BASENAME + str(tag).zfill(4)),
                                    source="flash") for tag in file_tags]
    return datasets


def process_dataset(dataset):
    """
    Process dataset to get values for benchmark quantities
    """
    merged_dataset = boxkit.mergeblocks(dataset, ["dfun", "velx", "vely"])
    merged_dataset.fill_guard_cells()

    shapelist = flash_box.lset_shape_measurement_2d(merged_dataset, correction=True)
    quantlist = flash_box.lset_quant_measurement_2d(merged_dataset)

    if len(shapelist) != len(quantlist):
        raise ValueError(f"len(shapelist) == {len(shapelist)} and len(quantlist) == {len(quantlist)}")

    max_bubble_area = 1e-13
    main_bubble_shape = None
    main_bubble_quant = None

    for shape, quant in zip(shapelist, quantlist):
        if shape["area"] > max_bubble_area:
            main_bubble_shape = shape
            main_bubble_quant = quant

    circularity = (2*numpy.pi*numpy.sqrt(main_bubble_shape["area"]/numpy.pi)/main_bubble_shape["perimeter"])
    center = main_bubble_quant["centroid"][0] - SIM_YMIN
    area = main_bubble_shape["area"]
    velocity = main_bubble_quant["velocity"][0]
    time = dataset.time

    return numpy.array([float(time), float(area), float(circularity), float(center), float(velocity)])

def compute_norm(fine_datasets, coarse_datasets, order=None, scale=None):
    """
    Compute norm between two datasets
    """
    varlist = ["dfun", "velx", "vely"]

    fine_blocks =  [boxkit.mergeblocks(dataset, varlist).blocklist[0] for dataset in fine_datasets]
    coarse_blocks = [boxkit.mergeblocks(dataset, varlist).blocklist[0] for dataset in coarse_datasets]

    norm = dict(zip(varlist, [0.]*len(varlist)))

    for fblock, cblock in zip(fine_blocks, coarse_blocks):

        if not scale:
            scale = {"coarse": 1, "fine": int(cblock.dx/fblock.dx)}

        for var in varlist:
            cdata = cblock[var][0,cblock.yguard:cblock.nyb+cblock.yguard, cblock.xguard:cblock.nxb+cblock.xguard]        
            fdata = fblock[var][0,fblock.yguard:fblock.nyb+fblock.yguard, fblock.xguard:fblock.nxb+fblock.xguard]

            cwork = get_offset_data(cdata, scale["coarse"])
            fwork = get_offset_data(fdata, scale["fine"])

            norm[var] = norm[var] + numpy.linalg.norm(fwork-cwork, ord=order)/len(coarse_blocks)

    return norm

def get_offset_data(data, scale):
    """
    Offset data
    """
    if scale == 1:
        work = data
    else:
        work = 0.
        for xoff in range(scale):
            for yoff in range(scale):
                work = work + data[yoff::scale, xoff::scale]/(scale**2)

    return work

def case2_grid_convergence_dict():
    """
    Get dictionary for grid independence study
    """
    dataset_dir = {}
    dataset_dir["Case2/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h320/jobnode.archive/2023-11-11"

    file_tags = {}
    file_tags["Case2/h40"] = [43]
    file_tags["Case2/h80"] = [43]
    file_tags["Case2/h160"] = [43]
    file_tags["Case2/h320"] = [53]

    return dataset_dir, file_tags


def case2_time_convergence_dict():
    """
    Get dictionary for grid independence study
    """
    dataset_dir = {}
    dataset_dir["Case2/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h320/jobnode.archive/2023-11-11"

    file_tags = {}
    file_tags["Case2/h40"] = [*range(40)]
    file_tags["Case2/h80"] = [*range(40)]
    file_tags["Case2/h160"] = [*range(40)]
    file_tags["Case2/h320"] = [*range(62)]

    return dataset_dir, file_tags


def case1_time_convergence_dict():
    """
    Get dictionary for grid independence study
    """
    dataset_dir = {}
    dataset_dir["Case1/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h160/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h320/jobnode.archive/2023-11-08"

    file_tags = {}
    file_tags["Case1/h40"] = [*range(51)]
    file_tags["Case1/h80"] = [*range(51)]
    file_tags["Case1/h160"] = [*range(51)]
    file_tags["Case1/h320"] = [*range(51)]

    return dataset_dir, file_tags


def case2_refinement_contour_dict():
    """
    Get dictionary to compare bubble contours for case 2
    """
    dataset_dir = {}
    dataset_dir["Case2/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h320/jobnode.archive/2023-11-11"

    file_tags = {}
    file_tags["Case2/h40"] = [0,15,29,43,49]
    file_tags["Case2/h80"] = [0,15,29,43,49]
    file_tags["Case2/h160"] = [0,15,29,43,49]
    file_tags["Case2/h320"] = [0,18,36,53,60]

    return dataset_dir, file_tags


def case2_outflow_contour_dict():
    """
    Get dictionary to compare bubble contours for case 2
    """
    dataset_dir = {}
    dataset_dir["Case2/h160/lb0.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_0.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.0"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.0_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"

    file_tags = {}
    file_tags["Case2/h160/lb0.5"] = [0,16,30,45,51]
    file_tags["Case2/h160/lb1.0"] = [0,15,30,45,51]
    file_tags["Case2/h160/lb1.5"] = [0,16,30,45,51]
    file_tags["Case2/h160"] = [0,15,29,43,49]

    return dataset_dir, file_tags


def case2_width_dict():
    """
    Get dictionary for gird refinement study for case 2
    """
    dataset_dir = {}
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h160-lx-3"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160-lx-3/jobnode.archive/2023-11-21"

    file_tags = {}
    file_tags["Case2/h160"] = [*range(51)]
    file_tags["Case2/h160-lx-3"] = [*range(51)]

    return dataset_dir, file_tags


def case2_refinement_dict():
    """
    Get dictionary for gird refinement study for case 2
    """
    dataset_dir = {}
    dataset_dir["Case2/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h320/jobnode.archive/2023-11-11"

    file_tags = {}
    file_tags["Case2/h40"] = [*range(51)]
    file_tags["Case2/h80"] = [*range(51)]
    file_tags["Case2/h160"] = [*range(51)]
    file_tags["Case2/h320"] = [*range(61)]

    return dataset_dir, file_tags


def case1_refinement_dict():
    """
    Get dictionary for gird refinement study for case 1
    """
    dataset_dir = {}
    dataset_dir["Case1/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h160/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h320/jobnode.archive/2023-11-08"

    file_tags = {}
    file_tags["Case1/h40"] = [*range(51)]
    file_tags["Case1/h80"] = [*range(51)]
    file_tags["Case1/h160"] = [*range(51)]
    file_tags["Case1/h320"] = [*range(51)]

    return dataset_dir, file_tags


def case2_outflow_dict():
    """
    Get dictionary for outflow study for case 2
    """
    dataset_dir = {}
    dataset_dir["Case2/h160/lb0.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_0.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.0"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.0_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"

    file_tags = {}
    file_tags["Case2/h160/lb0.5"] = [*range(53)]
    file_tags["Case2/h160/lb1.0"] = [*range(53)]
    file_tags["Case2/h160/lb1.5"] = [*range(53)]
    file_tags["Case2/h160"] = [*range(51)]

    return dataset_dir, file_tags


if __name__ == "__main__":
    """
    Main
    """
    pass
