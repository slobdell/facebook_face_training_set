import cv
import cv2
import os
import Image
import json
import numpy as np
import urllib2
from StringIO import StringIO

CASCADE = "./haarcascade_frontalface_alt.xml"
OUTPUT_DIRECTORY = "./face_root_directory/"
ACCESS_TOKEN = "YOUR FACEBOOK API TOKEN GOES HERE.  Visit https://developers.facebook.com/tools/explorer for easy access to a token"

IMAGE_SCALE = 2
haar_scale = 1.2
min_neighbors = 3
min_size = (20, 20)
haar_flags = 0
normalized_face_dimensions = (100, 100)

FB_BASE = "https://graph.facebook.com/"


def make_request(query):
    url = "%s%s" % (FB_BASE, query)
    request = urllib2.Request(url)
    try:
        response = urllib2.urlopen(request)
    except urllib2.HTTPError:
        print "Error on URL: %s" % url
        return []
    response_text = response.read()
    response_dict = json.loads(response_text)
    return response_dict.get("data", [])


def get_friend_service_ids():
    query = "me/friends?limit=5000&offset=0&access_token=%s" % ACCESS_TOKEN
    facebook_response = make_request(query)
    all_friends = facebook_response
    service_ids = [friend.get("id") for friend in all_friends]
    return service_ids


def get_all_photos(service_id):
    query = "%s/photos?type=tagged&limit=5000&access_token=%s" % (service_id, ACCESS_TOKEN)
    return make_request(query)


def get_name(service_id):
    query = "%s?access_token=%s" % (service_id, ACCESS_TOKEN)

    url = "%s%s" % (FB_BASE, query)
    request = urllib2.Request(url)
    try:
        response = urllib2.urlopen(request)
    except urllib2.HTTPError:
        "Error getting name for %s" % url
        return None
    response_text = response.read()
    response_dict = json.loads(response_text)
    name = response_dict.get("name")
    return name


def convert_rgb_to_bgr(open_cv_image):
    try:
        new_image = cv.CreateImage((open_cv_image.width, open_cv_image.height), cv.IPL_DEPTH_8U, open_cv_image.channels)
        cv.CvtColor(open_cv_image, new_image, cv.CV_RGB2BGR)
    except:
        print "Error converting image to BGR"
        return None
    return new_image


def download_photo_as_open_cv_image(photo_url):
    try:
        img = urllib2.urlopen(photo_url).read()
    except urllib2.HTTPError:
        # possible case of 404 on image
        print "Error fetching image: %s" % photo_url
        return None
    img = StringIO(img)
    pil_image = Image.open(img)
    try:
        open_cv_image = cv.fromarray(np.array(pil_image))[:, :]
    except TypeError:
        print "unsupported image type"
        return None
    open_cv_image = convert_rgb_to_bgr(open_cv_image)
    return open_cv_image


def normalize_image_for_face_detection(img):
    gray = cv.CreateImage((img.width, img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / IMAGE_SCALE),
                   cv.Round(img.height / IMAGE_SCALE)), 8, 1)
    if img.channels > 1:
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray = cv.CloneMat(img[:, :])
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)
    cv.EqualizeHist(small_img, small_img)
    return small_img


def _is_in_bounds(region_of_interest, constraint_coordinate, open_cv_image):
    '''
    Region of interest: (x, y, w, h)
    Constraint Coordinate: x and y as a percent from left and top
    '''
    constraint_coordinate = (constraint_coordinate[0] / 100.0, constraint_coordinate[1] / 100.0)
    translated_coordinate = (open_cv_image.width * constraint_coordinate[0], open_cv_image.height * constraint_coordinate[1])
    x_min = region_of_interest[0]
    x_max = region_of_interest[2] + x_min
    y_min = region_of_interest[1]
    y_max = region_of_interest[3] + y_min

    if x_min <= translated_coordinate[0] <= x_max and y_min <= translated_coordinate[1] <= y_max:
        return True
    return False


def normalize_face_size(face):
    normalized_face_dimensions = (100, 100)
    face_as_array = np.asarray(face)
    resized_face = cv2.resize(face_as_array, normalized_face_dimensions)
    resized_face = cv.fromarray(resized_face)
    return resized_face


def normalize_face_histogram(face):
    face_as_array = np.asarray(face)
    equalized_face = cv2.equalizeHist(face_as_array)
    equalized_face = cv.fromarray(equalized_face)
    return equalized_face


def normalize_face_color(face):
    gray_face = cv.CreateImage((face.width, face.height), 8, 1)
    if face.channels > 1:
        cv.CvtColor(face, gray_face, cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray_face = cv.CloneMat(face[:, :])
    return gray_face[:, :]


def normalize_face_for_save(face):
    face = normalize_face_size(face)
    face = normalize_face_color(face)
    face = normalize_face_histogram(face)
    return face


def face_detect_on_photo(img, constraint_coordinate):
    cascade = cv.Load(CASCADE)
    faces = []
    small_img = normalize_image_for_face_detection(img)
    faces_coords = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                        haar_scale, min_neighbors, haar_flags, min_size)
    for ((x, y, w, h), n) in faces_coords:
        if constraint_coordinate is not None and not _is_in_bounds((x, y, w, h), constraint_coordinate, small_img):
            continue
        pt1 = (int(x * IMAGE_SCALE), int(y * IMAGE_SCALE))
        pt2 = (int((x + w) * IMAGE_SCALE), int((y + h) * IMAGE_SCALE))
        face = img[pt1[1]:pt2[1], pt1[0]: pt2[0]]
        face = normalize_face_for_save(face)
        faces.append(face)
    return faces


#  @task
def get_face_in_photo(photo_url, service_id, picture_name, name, x, y):
    photo_in_memory = download_photo_as_open_cv_image(photo_url)

    # TODO: make the network call asynchronous as well with a callback function
    if photo_in_memory is None:
        return
    if x is None and y is None:
        # case for profile picture that isnt necessarily tagged
        # only return a result if exactly one face is in the image
        faces = face_detect_on_photo(photo_in_memory, None)
        if len(faces) == 1:
            save_face(name, service_id, faces[0], picture_name)
        return
    for face in face_detect_on_photo(photo_in_memory, (x, y)):
        save_face(name, service_id, face, picture_name)


# @task
def get_tagged_photos(service_id, name):
    all_photos = get_all_photos(service_id)
    picture_count = 0
    for photo in all_photos:
        photo_url = photo.get("source")
        tag_list = photo.get("tags", {}).get("data")
        if tag_list is None:
            continue
        for tag in tag_list:
            if tag.get("name") == name:
                # X and Y are the percentages from top and left
                # need to
                x = float(tag.get("x", 100))
                y = float(tag.get("y", 100))
                if 100 not in (x, y):
                    get_face_in_photo(photo_url, service_id, picture_count, name, x, y)  # TODO: apply asynchronously
                    picture_count += 1


def get_profile_picture_album_id(service_id):
    query = "%s/albums?limit=5000&access_token=%s" % (service_id, ACCESS_TOKEN)
    all_albums = make_request(query)
    album_id = None
    for album in all_albums:
        if album.get("name") == "Profile Pictures":
            album_id = album.get("id")
            break
    return album_id


#  @task
def get_profile_photos(service_id, name):
    album_id = get_profile_picture_album_id(service_id)
    if album_id is None:
        return
    query = "%s/photos?limit=5000&access_token=%s" % (album_id, ACCESS_TOKEN)
    all_photos = make_request(query)
    picture_count = 0
    for photo in all_photos:
        photo_url = photo.get("source")
        picture_name = "profile_%s" % picture_count
        get_face_in_photo(photo_url, service_id, picture_name, name, None, None)
        picture_count += 1


def _create_folder_name(name, service_id):
    split_name = name.split(" ")
    first = split_name[0]
    last = split_name[len(split_name) - 1]
    folder_name = "%s_%s_%s" % (last, first, service_id)
    return folder_name


def save_face(name, service_id, face, picture_name):
    folder_name = _create_folder_name(name, service_id)
    if not os.path.exists(OUTPUT_DIRECTORY + folder_name):
        os.makedirs(OUTPUT_DIRECTORY + folder_name)
    filename = "%s.jpg" % picture_name
    full_path = OUTPUT_DIRECTORY + folder_name + "/" + filename
    try:
        cv2.imwrite(full_path, np.asarray(face))
    except UnicodeEncodeError:
        print "Did not save picture because of unicode exception"
        # TODO: Use django smart_bytes to save the image here
    print "Saving: %s" % full_path


if __name__ == "__main__":
    service_ids = get_friend_service_ids()
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    for service_id in service_ids:
        name = get_name(service_id)
        if name is None:
            continue
        get_profile_photos(service_id, name)  # TODO: apply asynchronously
        get_tagged_photos(service_id, name)  # TODO: apply_asyncchronously
