'''
For new cards
(x,y)    top left  bottom right
Round 2: (297,421) (400,600)
Round 3: (440,423) (548,601)
Round 4: (583,421) (700,600)
'''

import cv2
import numpy as np
import glob

def count_holes(binary_img):
    contours, hierarchy = cv2.findContours(
        binary_img,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return 0

    holes = 0
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent != -1:  # has parent → hole
            area = cv2.contourArea(contours[i])
            if area > 5:   # ignore noise
                holes += 1
    return holes


def find_rank(screenshot_path, template_paths,x_start,x_end,y_start,y_end):
    image = cv2.imread(screenshot_path)
    roi = image[y_start:y_end,x_start:x_end]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Cropped Gray Image

    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)


    best_match = None
    best_score =  -1
    for template_path in template_paths:
        template = cv2.imread(template_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(
            roi_gray,
            template,
            cv2.TM_CCOEFF_NORMED
        )
        threshold = 0.8
        score_gray = float(result.max())
        score = score_gray

        if any(r in template_path.lower() for r in ["q", "8"]):

            _, roi_bin = cv2.threshold(
                roi_gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            holes = count_holes(roi_bin)

            if holes == 2:
                score += 0.15
                if "8" in template_path.lower():
                    score += 0.2
            elif holes == 1:
                score += 0.15
                if "q" in template_path.lower():
                    score += 0.2

        if score > best_score:
            best_score = score
            best_match = template_path
    print("Best Match: " + best_match, "Accuracy: " + str(best_score))
    return template_path

def find_suit(screenshot_path,x_start,x_end,y_start,y_end):
    image = cv2.imread(screenshot_path)
    roi = image[y_start:y_end, x_start:x_end]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = (
        cv2.inRange(hsv, lower_red1, upper_red1) |
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # Black mask (low V)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    red_pixels = cv2.countNonZero(red_mask)
    black_pixels = cv2.countNonZero(black_mask)

    # --- Determine color ---
    is_red = red_pixels > black_pixels

    # --- Binary for shape analysis ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [c for c in contours if cv2.contourArea(c) > 20]

    if not contours:
        return None

    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    perimeter = cv2.arcLength(main, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    hull = cv2.convexHull(main)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    if is_red:
        if solidity > 0.75:
            suit = "hearts"
        else:
            suit = "diamonds"
    else:

        if circularity > 0.5:
            suit = "clubs"
        else:
            suit = "spades"


    # cv2.imshow("Suit ROI", roi)
    # cv2.waitKey(0)

    return suit



if __name__ == "__main__":
    # Round 2
    template_paths = glob.glob("../playing_card_examples/better_examples_cropped/*.png")
    print("=" * 5 + " Round 2 " + "=" * 5)
    screenshot_path = "../examples/Round2.png"
    image = cv2.imread(screenshot_path)

    # rank
    x_start = 297
    x_end = 311
    y_start = 429
    y_end = 451

    # suit
    x_start_suit = 292
    x_end_suit = 308
    y_start_suit = 450
    y_end_suit = 468
    new_image = image[y_start:y_end,x_start:x_end]
    cv2.imshow("Cropped Image", new_image)
    find_rank(screenshot_path,template_paths,x_start,x_end,y_start,y_end)
    print(find_suit(screenshot_path, x_start_suit, x_end_suit, y_start_suit, y_end_suit))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print("=" * 5 + " Round 3 " + "=" * 5)
    screenshot_path = "../examples/Round3.png"
    image = cv2.imread(screenshot_path)

    # rank
    x_start = 441
    x_end = 455
    y_start = 429
    y_end = 451

    # suit
    x_start_suit = 439
    x_end_suit = 453
    y_start_suit = 450
    y_end_suit = 468

    new_image = image[y_start:y_end,x_start:x_end]
    img_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Cropped Image", new_image)
    find_rank(screenshot_path,template_paths,x_start,x_end,y_start,y_end)
    print(find_suit(screenshot_path, x_start_suit, x_end_suit, y_start_suit, y_end_suit))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("=" * 5 + " Round 4 " + "=" * 5)
    screenshot_path = "../examples/Round4.png"
    image = cv2.imread(screenshot_path)

    x_start = 585
    x_end = 650
    y_start = 400
    y_end = 500

    # rank
    x_start = 585
    x_end = 601
    y_start = 430
    y_end = 452

    # suit
    x_start_suit = 581
    x_end_suit = 598
    y_start_suit = 451
    y_end_suit = 467
    new_image = image[y_start:y_end,x_start:x_end]
    cv2.imshow("Cropped Image", new_image)
    find_rank(screenshot_path,template_paths,x_start,x_end,y_start,y_end)

    print(find_suit(screenshot_path, x_start_suit, x_end_suit, y_start_suit, y_end_suit))
    cv2.waitKey(0)
    cv2.destroyAllWindows()