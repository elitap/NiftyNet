import argparse
import SimpleITK as sitk
import numpy as np
import os


ST1_CONFIG = {"full": ("", [96]), "half": ("_half", [96, 48]), "quarter": ("_quarter", [48])}
ST2_CONFIG = {"full": ("", [48, 24])}

def get_ratio(itk_img, kernel_size):

    np_img = sitk.GetArrayFromImage(itk_img)
    np_img[np_img > 0] = 1

    itk_img = sitk.GetImageFromArray(np_img)

    cast_img_filter = sitk.CastImageFilter()
    cast_img_filter.SetOutputPixelType(sitk.sitkUInt8)
    itk_img = cast_img_filter.Execute(itk_img)

    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelType(sitk.sitkBox)
    dilation.SetKernelRadius(kernel_size / 2)
    dilation.SetForegroundValue(1)
    dilation.SetBackgroundValue(0)
    itk_img = dilation.Execute(itk_img)

    np_img = sitk.GetArrayFromImage(itk_img)

    unique, count = np.unique(np_img.flatten(), return_counts=True)
    return float(count[unique>0].sum()) / float(np_img.flatten().size)


def get_foreground_ratio(datasetpath, result_file, filter, config):
    overall_ratios = []
    with open(result_file, 'w') as fileptr:
        fileptr.write("CT_size,Window_size,Ratio\n")
        for key, value in config.iteritems():

            datasetpostfix, window_sizes = value
            path = (datasetpath % datasetpostfix)

            for window_size in window_sizes:
                ratios = []

                for file in os.listdir(path):
                    if filter in file:
                        itk_img = sitk.ReadImage(os.path.join(path, file))
                        ratio = get_ratio(itk_img, window_size)
                        print ratio
                        ratios.append(ratio)
                        overall_ratios.append(ratio)

                print key, window_size, np.array(ratios).mean()
                fileptr.write(key + "," + str(window_size) + "," + str(np.array(ratios).mean()) + "\n")

    print "Ratio", np.array(overall_ratios).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--datasetpath',
                        required=True,
                        help="dataset base path"
                        )
    parser.add_argument('--resultfile',
                        required=True,
                        help="path to the result csv"
                        )
    parser.add_argument('--dataset',
                        required=False,
                        choices=['Train', 'Test'],
                        default='Test',
                        help="dataset"
                        )
    parser.add_argument('--filter',
                        required=False,
                        type=str,
                        default='foreground',
                        help="identifier for the label"
                        )
    parser.add_argument('--stage',
                        required=False,
                        choices=['coarse', 'fine'],
                        default='coarse'
                        )

    args = parser.parse_args()
    path = os.path.join(args.datasetpath, args.dataset) + "%s"

    config = None
    if args.stage == 'coarse':
        config = ST1_CONFIG
    if args.stage == 'fine':
        config = ST2_CONFIG

    get_foreground_ratio(path, args.resultfile, args.filter, config)
