import argparse
import SimpleITK as sitk
import numpy as np
import os

 
VALID_FILES = [".mha", ".nrrd", ".mhd"]

class ImageInfo:
    def __init__(self, file, size, spacing, minMax, labelBB = None):
        self.file = file
        self.spacing = spacing
        self.size = size
        self.minMax = minMax
        self.labelBB = labelBB


def getImageInfo(filename, datasetpath):
    if os.path.splitext(filename)[1] in VALID_FILES:
        full_file = datasetpath + "/" + filename
        itk_img = sitk.ReadImage(full_file)
        img = sitk.GetArrayFromImage(itk_img)
        return itk_img, ImageInfo(full_file, itk_img.GetSize(), itk_img.GetSpacing(), [int(img.min()), int(img.max())])
    else:
        return None, None


def getBBforLabelId(itk_img, labelId):
    labelShapesFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapesFilter.Execute(itk_img)
    return labelShapesFilter.GetBoundingBox(labelId)  # get bb of segmentation

 
def collectImgWithLabels(dataset, labelpostfix, volumepostfix):
 
    dataset = os.path.abspath(dataset)
    imageInfoList = []

    # for test purposes just use some of the files in the dir eg
    # os.listdir(dataset)[:6]
    for file in os.listdir(dataset):

        if volumepostfix in file:
            _, volumeInfo = getImageInfo(file, dataset)

            # find the corresponding segmentation
            segmentationInfo = None
            foregroundInfo = None
            for segmentation in os.listdir(dataset):
                if labelpostfix in segmentation and file[:9] in segmentation:
                    itk_img, segmentationInfo = getImageInfo(segmentation, dataset)

                    #mand_and_par = sitk.BinaryThreshold(itk_img, 5, 8) #lables for right/left parotis and mandibula

                    #segmentationInfo.labelBB = getBBforLabelId(mand_and_par, 1) #get bb of segmentation

                #if "foreground" in segmentation and file[:9] in segmentation:
                #    _, foregroundInfo = getImageInfo(segmentation, dataset)

            if volumeInfo is not None and segmentationInfo is not None:# and foregroundInfo is not None:
                imageInfoList.append((volumeInfo,segmentationInfo)) #append a tuple (volume, segmentation)
            else:
                print("no segmentation or foreground found for: ", file)

    return imageInfoList
 
 
def callculateStatistics(imageInfoList):

    label_cnt = 0
    size = np.zeros([len(imageInfoList),3])
    spacing = np.zeros([len(imageInfoList),3])
    labelbb = np.zeros([len(imageInfoList),3])
    labelMax = np.zeros([len(imageInfoList),1])
 
    for _ ,segmentationInfo in imageInfoList:
        size[label_cnt,:] = segmentationInfo.size
        spacing[label_cnt, :] = segmentationInfo.spacing
        labelbb[label_cnt, :] = segmentationInfo.labelBB[3:]
        labelMax[label_cnt] = segmentationInfo.minMax[1]
        label_cnt += 1

    medianSpacing = np.around(np.median(spacing,0),2)
    meanSpacing = np.around(np.mean(spacing,0),2)
    print("Volume      size:  max -> x,y,z: ", size.max(0), " min -> x,y,z: ", size.min(0), " avg -> x,y,z: ", size.mean(0), " median -> x,y,z: ", np.median(size,0))
    print("BoundingBox size:  max -> x,y,z: ", labelbb.max(0), " min -> x,y,z: ", labelbb.min(0), " avg -> x,y,z: ", labelbb.mean(0), " median -> x,y,z: ", np.median(labelbb,0))
    print("Volume   spacing:  max -> x,y,z: ", spacing.max(0), " min -> x,y,z: ", spacing.min(0), " avg -> x,y,z: ", meanSpacing, " medain -> x,y,z: ", medianSpacing)

    print("\n")
    scale = spacing / medianSpacing
    print("Unify spacing to -> x,y,z ", medianSpacing)
    scaledLabelbb = labelbb * scale
    print("Unified BoundingBox size:  max -> x,y,z: ", scaledLabelbb.max(0), " min -> x,y,z: ", scaledLabelbb.min(0), " avg -> x,y,z: ",
         scaledLabelbb.mean(0), " median -> x,y,z: ", np.median(scaledLabelbb, 0))

    #boxPlot(spacing, "spacing")
    #boxPlot(size, "dimensions")
    #boxPlot(scaledLabelbb, "unified bb dimensions")

    return medianSpacing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')
 
    parser.add_argument('--dataset',
                        required=True,
                        help="dataset path containing ct image volumes"
                        )
    parser.add_argument('--labelfilter',
                        required=False,
                        default='segmentation',
                        help="name identifying label volumes in the given dataset dir"
                        )
    parser.add_argument('--imagefiler',
                        required=False,
                        default='volume',
                        help="name identifying image volumes in the given dataset dir"
                        )
 
    args = parser.parse_args()
    imageInfoList = collectImgWithLabels(args.dataset, args.labelfilter, args.imagefiler)
    callculateStatistics(imageInfoList)
