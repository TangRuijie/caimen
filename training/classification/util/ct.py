import SimpleITK as sitk
import pydicom
import time

def dcmread(p, read_type = 'pydicom', wait_count = 5):
    ds = None
    gray = None
    while True:
        try:
            if read_type == 'pydicom':
                ds = pydicom.dcmread(p)
                gray = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            else:
                itk_img = sitk.ReadImage(p)
                gray = sitk.GetArrayFromImage(itk_img)[0]
            break
        except:
            time.sleep(5)
            wait_count -= 1
            if wait_count == 0:
                break
    return ds, gray
