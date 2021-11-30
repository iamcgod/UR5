import os
import cv2
import time
import numpy as np
import imutils
from utils import pytorch_utility
import time

def predict_num(cvimg):
    t_start = time.perf_counter()
    model = pytorch_utility.load_full_model(filename="./resnet18_sgd_00005_0907_num_2.pth")
    t_end = time.perf_counter()
    print(type(model), "Loading time: ", round((t_end- t_start), 3), " sec.")
    tag, pro = pytorch_utility.predict_cvImg(model, "cuda:0", cvimg, resize_height=224, resize_width =224, isShow=False, isStdNormalized =True)
    return tag


def path_to_fname(file_path):
    fname = file_path.rsplit("/", 1)[-1]
    fname = fname.rsplit("\\", 1)[-1]
    return fname


def img_imread_BGR(img_path):
    img = cv2.imread(img_path, 1)
    img_h, img_w = img.shape[:2]
    img_shape_hw = (img_h, img_w)
    return img, img_shape_hw

def img_BGR(img):
    img_h, img_w = img.shape[:2]
    img_shape_hw = (img_h, img_w)
    return img, img_shape_hw

def template_match_scale_single(img, template, method=1, minScale=0.95, maxScale=1.05, scale_step=0.01,  isShow=False):
    """minScale, maxScale: 將影像放大縮小的範圍"""
    t_start = time.perf_counter()
    if len(img.shape) == 3:
        img_c = img.copy()
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    else:
        img = img.copy()
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_w, img_h = img.shape[::-1]
    template_c = template.copy()
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template = template.copy()
    temp_w, temp_h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
               'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
    # print("Matching method", methods[method])
    m = eval(methods[method])
    max = 0  # max matching score
    # res_list = list()
    max_res = None
    # find the most similar scale
    for scale in np.arange(minScale, maxScale+scale_step/2, scale_step):
        temp = cv2.resize(template.copy(), None, fx=scale, fy=scale)
        # print(temp.shape)
        best_temp_w, best_temp_h = temp.shape[::-1]
        if temp.shape[0] > img_h or temp.shape[1] > img_w:
            break
        res = cv2.matchTemplate(img, temp, method)  # 顯示左上角的點，並非中心
        real_res = res.copy()
        if m <= 1:  # 'cv2.TM_SQDIFF_NORMED': min is matched & between 0 and 1
            res = 0 - res

        while True:  # 避免template超出圖片邊界
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_loc[0]+best_temp_w > img_w or max_loc[1]+best_temp_h > img_h:
                res[max_loc[1]][max_loc[0]] = 0
                if m <= 1:  # 越小越匹配
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
                        real_res)
                    real_res[max_loc[1]][max_loc[0]] = max_val
                else:  # 越大越匹配
                    real_res[max_loc[1]][max_loc[0]] = 0
            else:
                break
        # 找最大值
        if max_val > max or max == 0:
            max = max_val
            max_real_res = real_res.copy()
            best_scale = scale
            best_temp = temp
        # res_list.append(res)
    if max == 0:  # 這個旋轉角度的圖塞不下
        return None, None, None, None, None, None
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(max_real_res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  # min is matched
        top_left = min_loc
        value = min_val
    else:
        top_left = max_loc
        value = max_val
    best_temp_w, best_temp_h = best_temp.shape[::-1]
    bottom_right = (top_left[0] + best_temp_w, top_left[1] + best_temp_h)
    pt = [top_left, bottom_right]

    cv2.rectangle(img_c, top_left, bottom_right, (0, 0, 255), 5)
    t_end = time.perf_counter()
    time_spend = round((t_end - t_start), 3)
    if isShow:
        print("Matching Time: ", time_spend, " sec.")
        cv2.imshow(methods[m] + " Match Result:", img_c)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_c, best_temp, pt, value, best_scale, time_spend


def find_obj_area(img, isShow=False):
    ## img二值化 ##
    img_h, img_w = img.shape[0], img.shape[1]  # H,W,C
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bgr = img.copy()
    else:
        img_gray = img.copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ret, img_thresh = cv2.threshold(img_gray, 127, 255, 0)
    ## show img & threshold ##
    # cv2.namedWindow("img_gray", cv2.WINDOW_NORMAL)
    # cv2.imshow("img_gray", img_gray)
    # cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
    # cv2.imshow("threshold", img_thresh)
    # cv2.waitKey(0)
    ## 找物件的contour ##
    contours, hierarchy,_ = cv2.findContours(
        img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("len(contours): ", len(contours))
    max_area = 0
    side_border = min(int(img_w/6), int(img_h/6))
    while True:
        for n in range(len(contours)):
            box = cv2.boundingRect(contours[n])  # (x,y,w,h)
            # print("contour", n, " area: ", box)
            x, y, w, h = box
            ## 淘汰邊界contours ##
            max_contour_ratio = 0.75
            if w < max_contour_ratio*img_w or h < max_contour_ratio*img_h:
                if x > side_border and y > side_border and x+w < img_w-side_border and y+h < img_h-side_border:
                    if w*h > max_area:
                        item_box = box
                        max_area = w*h
        if max_area == 0:
            side_border = int(side_border/6)
        else:
            break
    x, y, w, h = item_box
    ## 盡量框到所有白色物件 ##
    while True:
        item_range = 100
        step = 1
        if 255 in img_thresh[y:y+h, x-item_range:x]:  # 左
            x, y, w, h = x-step, y, w+step, h
        elif 255 in img_thresh[y:y+h, x+w:x+w+item_range]:  # 右
            x, y, w, h = x+step, y, w+step, h
        elif 255 in img_thresh[y-item_range:y, x:x+w]:  # 上
            x, y, w, h = x, y-step, w, h+step
        elif 255 in img_thresh[y+h:y+h+item_range, x:x+w]:  # 下
            x, y, w, h = x, y+step, w, h+step
        else:
            break
    center_x = int(x+w/2)
    center_y = int(y+h/2)
    ## 彙整框內的所有contours ##
    item_contour = []
    for n in range(len(contours)):
        box2 = cv2.boundingRect(contours[n])  # (左上x,左上y,w,h)
        x2, y2, w2, h2 = box2
        if x2 >= x and y2 >= y:
            if x2+w2 <= x+w and y2+h2 <= y+h:
                item_contour.extend(contours[n])
    ## 繪製外接矩形 ##
    obj_box = cv2.boundingRect(np.array(item_contour))  # (左上x,左上y,w,h)
    obj_x, obj_y, obj_w, obj_h = obj_box
    y1, y2 = int(obj_y-0.5*obj_h), int(obj_y+1.5*obj_h)  # 高變兩倍
    x1, x2 = int(obj_x-0.5*obj_w), int(obj_x+1.5*obj_w)  # 寬變兩倍
    if w/h < 0.3:  # 高瘦
        x1, x2 = int(obj_x-1*obj_w), int(obj_x+2*obj_w)  # 寬變三倍
    if w/h > 3:  # 矮胖
        y1, y2 = int(obj_y-1*obj_h), int(obj_y+2*obj_h)  # 高變三倍
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0
    if y2 >= img_h:
        y2 = img_h-1
    if x2 >= img_w:
        x2 = img_w-1
    obj_img = img[y1:y2, x1:x2]
    obj_pt = [(x1, y1), (x2, y2)]
    if isShow:
        obj_area_img = cv2.rectangle(
            img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 10)
        obj_area_img_s = cv2.resize(obj_area_img, (640, int(640*img_h/img_w)))
        cv2.imshow("obj_area_img", obj_area_img_s)
        cv2.waitKey(0)
    return obj_img, obj_pt


def make_golden(defect_img, pattern_path, angle_size=5, scale_size=0.05, scale_step=0.02):
    t_start = time.perf_counter()
    defect_img, defect_shape = img_BGR(defect_img)
    pattern_img, pattern_shape = img_imread_BGR(pattern_path)
    defect_img_1c = defect_img#cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    pattern_img_1c = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)
    (img_h, img_w) = defect_shape
    (imgP_h, imgP_w) = pattern_shape
    angle_size = int(angle_size)
    ## find obj area ##
    obj_img, obj_pt = find_obj_area(defect_img_1c, isShow=False)
    ## Patten Match ##
    max_value = float("-inf")  # 負無窮
    for angle in range(-angle_size, angle_size+1):
        # M = cv2.getRotationMatrix2D(
        #     (int(imgP_w/2), int(imgP_h/2)), angle, 1.0)
        # rotate_golden = cv2.warpAffine(pattern_img_1c, M, (imgP_w, imgP_h))
        rotate_golden = imutils.rotate_bound(pattern_img_1c, angle)
        method = 0
        img_draw, best_temp, pt, value, scale, time_spend = template_match_scale_single(
            obj_img, rotate_golden, method=method, minScale=1-scale_size, maxScale=1+scale_size,
            scale_step=scale_step, isShow=False)
        if scale == None:
            continue
        if method <= 0:
            value = 0 - value
        if value > max_value:
            max_value = value
            best_pt = pt  # [top_left, bottom_right]
            best_pattern = best_temp
            best_angle = angle
            best_scale = scale
            best_img = img_draw
    print("best_scale: ", best_scale)
    print("best_angle: ", best_angle)
    r_golden_sample = defect_img_1c.copy()
    r_golden_sample[obj_pt[0][1]+best_pt[0][1]:obj_pt[0][1]+best_pt[1][1],
                    obj_pt[0][0]+best_pt[0][0]:obj_pt[0][0]+best_pt[1][0]] = best_pattern
    best_pattern = cv2.cvtColor(best_pattern, cv2.COLOR_GRAY2BGR)
    t_end = time.perf_counter()
    time_spend = round((t_end - t_start), 3)
    print("total_time: ", time_spend)
    return best_pattern, r_golden_sample, best_img, best_angle, best_scale, time_spend


def img_diff(defect_img, golden_sample):
    if len(defect_img.shape) == 3:
        defect_img_1c = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    else:
        defect_img_1c = defect_img.copy()
    best_diff = cv2.absdiff(defect_img_1c, golden_sample)
    return best_diff


def opening(best_diff, kernel_size=5, iterations=2):
    kernel_size, iterations = int(kernel_size), int(iterations)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(
        best_diff, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening


def find_max_contour(feature_img, background_img, side_border=0):
    if len(feature_img.shape) == 3:
        feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)
    if len(background_img.shape) == 2:
        background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)
    img_h, img_w = feature_img.shape[:2]
    contours, hierarchy = cv2.findContours(
        feature_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    box_list = []
    contour_list = []
    for n in range(len(contours)):
        box = cv2.boundingRect(contours[n])  # (x,y,w,h)
        # print("contour", n, " area: ", box)
        x, y, w, h = box
        ## 淘汰邊界contours ##
        max_contour_ratio = 0.7
        if w < max_contour_ratio*img_w or h < max_contour_ratio*img_h:
            if x > side_border and y > side_border and x+w < img_w-side_border and y+h < img_h-side_border:
                # box_list.append(box)
                contour_list.append(contours[n])
    ## draw countours ##
    # for b in box_list: # 畫出瑕疵外接矩形
    #     cv2.drawContours(background_img, [b], 0, (0, 0, 255), 3)
    for c in contour_list:  # 畫出瑕疵邊框
        cv2.drawContours(background_img, [c], 0, (0, 0, 255), 3)
    return background_img


def save_predictImg(predict_img, save_root, num):
    '''save_root:資料夾路徑 defect_path:圖片路徑'''
    # defect_fname = path_to_fname(defect_path)
    save_path = os.path.join(save_root, "predict_" + str(num))
    cv2.imwrite(save_path, predict_img)
    print("save successfully!!")


def main_pyqt5(defect_path):
    ## defect_class: stain, scratch, zoom, mixed, mixedRT, incorrect, test ##
    golden_root = "./golden_sample_cropped"
    # defect_root = "./defective_pic/mixedRT"
    save_root = "./find_defect2/mixedRT"
    isShow = False
    isSave = False

    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    n = 1
    # defect_list = os.listdir(defect_root)
    # for defect_fname in defect_list:
    defect_fname = defect_path.rsplit("/", 1)[-1]
    defect_fname = defect_fname.rsplit("\\", 1)[-1]
    defect_name = defect_fname.rsplit(".", 1)[0]
    defect_img = cv2.imread(defect_path, 1)
    defect_img_1c = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = defect_img.shape[:2]
    golden_fname = defect_fname.split("_", 1)[-1]
    golden_path = os.path.join(golden_root, golden_fname)
    golden_img = cv2.imread(golden_path, 1)
    golden_img_1c = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    Gimg_h, Gimg_w = golden_img.shape[:2]

    ## Patten Match ##
    max_value = float("-inf")  # 負無窮
    for angle in range(-1, 2):
        # M = cv2.getRotationMatrix2D(
        #     (int(Gimg_w/2), int(Gimg_h/2)), angle, 1.0)
        # rotate_golden = cv2.warpAffine(golden_img_1c, M, (Gimg_w, Gimg_h))
        rotate_golden = imutils.rotate_bound(golden_img_1c, angle)
        method = 0
        img, best_temp, pt, value, scale, time_spend = template_match_scale_single(
            defect_img_1c, rotate_golden, method=method, minScale=0.8, maxScale=1.2, isShow=isShow)
        if scale == None:
            continue
        if method <= 0:
            value = 0 - value
        if value > max_value:
            max_value = value
            best_pt = pt  # [top_left, bottom_right]
            best_rotate_golden = best_temp
            best_angle = angle
            best_scale = scale
    print("best_scale: ", best_scale)
    print("best_angle: ", best_angle)
    r_golden_sample = defect_img_1c.copy()
    r_golden_sample[best_pt[0][1]:best_pt[1][1],
                    best_pt[0][0]:best_pt[1][0]] = best_rotate_golden
    best_diff = cv2.absdiff(defect_img_1c, r_golden_sample)

    ## find defect ##
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(
        best_diff, cv2.MORPH_OPEN, kernel, iterations=2)
    # side_border = int(np.min([y1, x1, img_h-y2, img_w-x2]))
    side_border = 0
    predict_img = find_max_contour(opening, defect_img, side_border)
    ## show img ##
    if isShow:
        best_diff = cv2.rectangle(
            best_diff, best_pt[0], best_pt[1], (0, 255, 0), 10)
        diff_r = cv2.resize(best_diff, None, fx=0.25, fy=0.25)
        cv2.imshow("diff_r", diff_r)
        cv2.waitKey(0)
        opening_r = cv2.resize(opening, None, fx=0.25, fy=0.25)
        cv2.imshow("opening", opening_r)
        cv2.waitKey(0)
        predict_img_r = cv2.resize(predict_img, None, fx=0.25, fy=0.25)
        cv2.imshow("predict_img", predict_img_r)
        cv2.waitKey(0)
    ## save img ##
    if isSave:
        save_path = os.path.join(save_root, "predict_"+defect_fname)
        cv2.imwrite(save_path, predict_img)
        # print(f"[{n}/{len(defect_list)}] is saving...")
    # n += 1
    print("finish!!")
    return predict_img
