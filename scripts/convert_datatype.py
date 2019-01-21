import argparse
import SimpleITK as sitk
import os





def convert(inputdir, outputdir, filefilter):
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    for file in os.listdir(inputdir):
        if filefilter in file:
            print "converting: ", file
            in_filepath = os.path.join(inputdir, file)
            itk_img = sitk.ReadImage(in_filepath, sitk.sitkUInt8)

            file_ending = os.path.splitext(file)[-1]
            out_filepath = os.path.join(outputdir, file.replace(file_ending,'.nii.gz'))
            sitk.WriteImage(itk_img, out_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--inputdir',
                        required=True
                        )
    parser.add_argument('--outputdir',
                        required=True
                        )
    parser.add_argument('--filefilter',
                        required=False,
                        default='segmentation'
                        )

    args = parser.parse_args()
    convert(args.inputdir, args.outputdir, args.filefilter)