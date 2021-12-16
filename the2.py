import cv2

def part1 ( input_img_path , output_path ) :
    img=cv2.imread( input_img_path)
    return img

def enhance_3 ( path_to_3 , output_path ) :
    img=cv2.imread( path_to_3)
    return img
    
def enhance_4 ( path_to_4 , output_path ) :
    img=cv2.imread( path_to_4)
    return img


def the2_write ( input_img_path , output_path ) :
    img=cv2.imread( input_img_path)
    img_name=""
    return img_name

def the2_read ( input_img_path ) :
    img=cv2.imread( input_img_path)
    cv2.imshow(img)