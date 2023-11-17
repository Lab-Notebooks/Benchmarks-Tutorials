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

    bubblelist = flash_box.lset_shape_measurement_2d(merged_dataset, correction=True)
    quantlist = flash_box.lset_quant_measurement_2d(merged_dataset)

    max_bubble_area = 1e-13
    main_bubble = None
    main_bubble_index = 0

    for index, bubble in enumerate(bubblelist):
        if bubble["area"] > max_bubble_area:
            main_bubble = bubble
            main_bubble_index = index

    circularity = (2*numpy.pi*numpy.sqrt(main_bubble["area"]/numpy.pi)/main_bubble["perimeter"])
    #center = main_bubble["centroid"][0] - SIM_YMIN
    center = quantlist[main_bubble_index]["centroid"][0] - SIM_YMIN
    area = main_bubble["area"]
    velocity = quantlist[main_bubble_index]["velocity"][0]
    time = dataset.time

    return numpy.array([float(time), float(area), float(circularity), float(center), float(velocity)])

def case2_refinement_contour_dict():
    """
    Get dictionary to compare bubble contours for case 2
    """
    dataset_dir = dict()
    dataset_dir["Case2/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h320/jobnode.archive/2023-11-11"

    file_tags = dict()
    file_tags["Case2/h40"] = [0,15,29,43,49]
    file_tags["Case2/h80"] = [0,15,29,43,49]
    file_tags["Case2/h160"] = [0,15,29,43,49]
    file_tags["Case2/h320"] = [0,18,36,53,60]

    return dataset_dir, file_tags

def case2_outflow_contour_dict():
    """
    Get dictionary to compare bubble contours for case 2
    """
    dataset_dir = dict()
    dataset_dir["Case2/h160/lb0.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_0.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.0"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.0_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"

    file_tags = dict()
    file_tags["Case2/h160/lb0.5"] = [0,16,30,45,51]
    file_tags["Case2/h160/lb1.0"] = [0,15,30,45,51]
    file_tags["Case2/h160/lb1.5"] = [0,16,30,45,51]
    file_tags["Case2/h160"] = [0,15,29,43,49]

    return dataset_dir, file_tags

def case2_refinement_dict():
    """
    Get dictionary for gird refinement study for case 2
    """
    dataset_dir = dict()
    dataset_dir["Case2/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"
    dataset_dir["Case2/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h320/jobnode.archive/2023-11-11"

    file_tags = dict()
    file_tags["Case2/h40"] = [*range(51)]
    file_tags["Case2/h80"] = [*range(51)]
    file_tags["Case2/h160"] = [*range(51)]
    file_tags["Case2/h320"] = [*range(61)]

    return dataset_dir, file_tags


def case1_refinement_dict():
    """
    Get dictionary for gird refinement study for case 1
    """
    dataset_dir = dict()
    dataset_dir["Case1/h40"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h40/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h80"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h80/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h160/jobnode.archive/2023-11-06"
    dataset_dir["Case1/h320"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case1/h320/jobnode.archive/2023-11-08"

    file_tags = dict()
    file_tags["Case1/h40"] = [*range(51)]
    file_tags["Case1/h80"] = [*range(51)]
    file_tags["Case1/h160"] = [*range(51)]
    file_tags["Case1/h320"] = [*range(51)]

    return dataset_dir, file_tags


def case2_outflow_dict():
    """
    Get dictionary for outflow study for case 2
    """
    dataset_dir = dict()
    dataset_dir["Case2/h160/lb0.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_0.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.0"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.0_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160/lb1.5"] = f"{SIM_PATH}/RisingBubble/OutflowTest/h160/buffer_1.5_growthRate_4.0/jobnode.archive/2023-11-08"
    dataset_dir["Case2/h160"] = f"{SIM_PATH}/RisingBubble/Benchmark/Case2/h160/jobnode.archive/2023-11-07"

    file_tags = dict()
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
