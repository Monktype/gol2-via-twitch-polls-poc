import pywinctl
import pyautogui
import cv2
import easyocr
import numpy as np
import time
import random
import requests
from typing import List, Dict, Any, Tuple, Optional

# Sorry, my formatting changed towards the end because a different, larger project
# needed different styling / typing, so there's a couple of different styles and
# added typing in this from different mindsets over a couple of weeks.
# It's a proof of concept that's put together with duct tape,
# so I'm not going to reformat anything right now.

### USER-DEFINED SETTINGS ###
# This OCR library is OK and it works without a ton of setup, but it's not perfect.
# Use at your own risk for what it will send to your Twitch channel, for example.
# "False" here will still run the library in some points (ie in "Choose your Opponent!")
# but will not print/send any outputs.
# (That dependency could be removed, but it's not a priority -- it needs an OpenCV scan to count player cards.)
use_ocr = False

# MSC location: Use this to specify where the MSC tool's API is running to get polls to Twitch.
msc_location = "http://localhost:8080"

# Fake polls with a delay and a random choice versus using MSC to create and monitor polls.
use_msc = False

# This is the numerical channel ID for the channel that the polls will be run.
# You can find this on Twitch, through Chatterino, or using `msc userid <channel name>`
channel_id = "0"


### LOADING STUFF ###

window_title = "Game Of Life 2"
wait_path = "./templates/WAIT.png"
templates_base_path = "./templates/"
spin_names = ["spin to move.png", "spin for your wedding gift.png", "spin to fulfill your bucket list.png", "spin the highest number.png", "spin to collect your pension ACTUAL SPIN.png", "spin to reveal your fate.png", "spin to collect.png"]
button_names = ["collect bonus.png", "fate spin result.png", "fate spin result money icon.png", "next highlighted.png", "ok highlighted.png", "take it highlighted.png", "take loan highlighted.png", "take loan.png", "sell houses button.png"]
money_button_names = ["collect bonus.png", "fate spin result.png", "fate spin result money icon.png"] # these are checked again with more sensitivity at the end of a check
jobs_names = ["choose a career job (floating some).png", "choose a college job (floating some).png", "choose a masters level job.png", "improve your job (floating some).png"]
choose_your_start_in_life_name = "choose your start in life.png"
which_way_to_go_name = "which way to go.png"
choose_an_action_name = "choose an action 1 (floating some).png"
choose_your_spouse_name = "choose your spouse three static options.png"
grow_your_family_name = "grow your family two static spaces.png"
where_to_retire_name = "where to retire two static buttons DIRTY.png"
choose_a_bucket_list_card_name = "choose a bucket list card (floating some).png"
select_your_opponent_name = "select your opponent.png"
choose_your_house_buysell_name = "choose your house buy sell.png"
choose_a_house_buy_name = "choose a house.png"
choose_a_house_sell_name = "choose a house to sell.png"

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# This takes forever to load relatively (like a second or so) and it's noisy in the logs
ocr_reader = easyocr.Reader(['en'], gpu=False)

wait_template = cv2.imread(wait_path)
gray_wait = cv2.cvtColor(wait_template, cv2.COLOR_BGR2GRAY)

gray_spins = []
for spin_name in spin_names:
    # gray_spins = [(filename string, gray image bin), ...]
    gray_spins.append((spin_name, cv2.cvtColor(cv2.imread(templates_base_path + spin_name), cv2.COLOR_BGR2GRAY)))

gray_buttons = []
for button_name in button_names:
    # gray_buttons = [(filename string, gray image bin), ...]
    gray_buttons.append((button_name, cv2.cvtColor(cv2.imread(templates_base_path + button_name), cv2.COLOR_BGR2GRAY)))

gray_money_buttons = []
for button_name in money_button_names:
    # gray_money_buttons = [(filename string, gray image bin), ...]
    gray_money_buttons.append((button_name, cv2.cvtColor(cv2.imread(templates_base_path + button_name), cv2.COLOR_BGR2GRAY)))

gray_jobs = []
for job_name in jobs_names:
    # gray_jobs = [(filename string, gray image bin), ...]
    gray_jobs.append((job_name, cv2.cvtColor(cv2.imread(templates_base_path + job_name), cv2.COLOR_BGR2GRAY)))

gray_choose_your_start_in_life = cv2.cvtColor(cv2.imread(templates_base_path + choose_your_start_in_life_name), cv2.COLOR_BGR2GRAY)

gray_which_way_to_go = cv2.cvtColor(cv2.imread(templates_base_path + which_way_to_go_name), cv2.COLOR_BGR2GRAY)

gray_choose_an_action = cv2.cvtColor(cv2.imread(templates_base_path + choose_an_action_name), cv2.COLOR_BGR2GRAY)

gray_choose_your_spouse = cv2.cvtColor(cv2.imread(templates_base_path + choose_your_spouse_name), cv2.COLOR_BGR2GRAY)

gray_grow_your_family = cv2.cvtColor(cv2.imread(templates_base_path + grow_your_family_name), cv2.COLOR_BGR2GRAY)

gray_where_to_retire = cv2.cvtColor(cv2.imread(templates_base_path + where_to_retire_name), cv2.COLOR_BGR2GRAY)

gray_choose_a_bucket_list_card = cv2.cvtColor(cv2.imread(templates_base_path + choose_a_bucket_list_card_name), cv2.COLOR_BGR2GRAY)

gray_select_your_opponent = cv2.cvtColor(cv2.imread(templates_base_path + select_your_opponent_name), cv2.COLOR_BGR2GRAY)

gray_choose_your_house_buysell = cv2.cvtColor(cv2.imread(templates_base_path + choose_your_house_buysell_name), cv2.COLOR_BGR2GRAY)

gray_choose_a_house_buy = cv2.cvtColor(cv2.imread(templates_base_path + choose_a_house_buy_name), cv2.COLOR_BGR2GRAY)

gray_choose_a_house_sell = cv2.cvtColor(cv2.imread(templates_base_path + choose_a_house_sell_name), cv2.COLOR_BGR2GRAY)



### SHARED / REUSABLE FUNCTIONS ###

# Determine if the cursor is in the control area
# (anywhere in the window except the top left corner)
# If it is, return True (a short to skip the rest of the processing).
# If the window isn't foreground, assume things won't work and short.
def cursor_short():
    window = pywinctl.getWindowsWithTitle(window_title)
    if not window:
        print("Window not found!")
        # Short control here, too.
        return True

    window = window[0]  # Get the first matching window... it should only be one for this case.

    if not window.isActive:
        # Assume things won't work, short.
        print("Window is not active, shorting automatic control.")
        return True

    # Get the window's position and size
    window_x, window_y, window_width, window_height = window.left, window.top, window.width, window.height

    # Get the current mouse position
    mouse_x, mouse_y = pyautogui.position()

    # Check if the cursor is within the window
    is_within_window = (window_x <= mouse_x <= window_x + window_width) and (window_y <= mouse_y <= window_y + window_height)

    if not is_within_window:
        # The window is foreground but the cursor is outside -- this is OK for automatic control
        return False

    # Define the relative proportions for the top left corner
    top_relative = 0.0  # From the top (0.0 means the top edge)
    bottom_relative = 0.85  # From the bottom
    left_relative = 0.0  # From the left (0.0 means the left edge)
    right_relative = 0.85  # From the right

    # Calculate the boundaries based on the relative proportions
    top_boundary = window_y + (window_height * top_relative)
    bottom_boundary = window_y + window_height - (window_height * bottom_relative)
    left_boundary = window_x + (window_width * left_relative)
    right_boundary = window_x + window_width - (window_width * right_relative)

    is_within_relative_region = (left_boundary <= mouse_x <= right_boundary) and (top_boundary <= mouse_y <= bottom_boundary)

    if not is_within_relative_region:
        print("Cursor is in control area, shorting automatic control.")
        return True

    return False


def crop_image(image, left_percent, right_percent, top_percent, bottom_percent):
    """
    Crop an image to a certain region specified by percentage.

    Args:
        image (numpy array): The input image.
        left_percent (float): The left boundary as a percentage of the image width.
        right_percent (float): The right boundary as a percentage of the image width.
        top_percent (float): The top boundary as a percentage of the image height.
        bottom_percent (float): The bottom boundary as a percentage of the image height.

    Returns:
        numpy array: The cropped image.
    """
    height, width, _ = image.shape
    left = int(width * left_percent / 100)
    right = int(width * (1 - right_percent / 100))
    top = int(height * top_percent / 100)
    bottom = int(height * (1 - bottom_percent / 100))

    return image[top:bottom, left:right]

def is_card(image, left_percent, right_percent, top_percent, bottom_percent):
    # Send a small region of the screenshot and this function tries to determine if this is a card background (the neutral offwhite color).
    # If it's a transparent card (like the "Sell" houses button when the player owns no houses), it should return false.

    cropped_image = crop_image(image, left_percent, right_percent, top_percent, bottom_percent)
    cropped_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 240])
    upper_bound = np.array([255, 20, 255])
    mask = cv2.inRange(cropped_hsv, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    mean_channels = cv2.mean(masked_image)[:3]
    mean_total = (mean_channels[0] + mean_channels[1] + mean_channels[2]) / 3

    # This can be a fairly liberal check due to the masking before.
    if mean_total >= 220:
        return True
    return False

def do_ocr_shared(image, left_percent, right_percent, top_percent, bottom_percent):
    # Don't use this function for checking direction, it's the shared function for the OCR components.
    ### Process the left and right options ###
    cropped = crop_image(image, left_percent, right_percent, top_percent, bottom_percent)
    # HSV is better for pre-OCR masking
    hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Mask to focus on card / neutral background (this is the offwhite background used in the game)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 25, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)

    gray_image = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR and formatting
    result = ocr_reader.readtext(binary_image)

    return result

def do_ocr(image, left_percent, right_percent, top_percent, bottom_percent):
    """
    This function runs OCR on a region of a screenshot.

    Args:
        image (numpy array): The input image.
        left_percent (float): The left boundary as a percentage of the image width.
        right_percent (float): The right boundary as a percentage of the image width.
        top_percent (float): The top boundary as a percentage of the image height.
        bottom_percent (float): The bottom boundary as a percentage of the image height.

    Returns:
        String: Output of the OCR, cleaned from known gotchas and bugs. (returned in all uppercase, YMMV if you don't)
    """
    result = do_ocr_shared(image, left_percent, right_percent, top_percent, bottom_percent)

    extracted_text = "\n".join([text[1].strip().lstrip() for text in result])
    extracted_text = extracted_text.replace("\n", " ")

    # These are common bugs with the OCR with the ! at the end of strings
    if extracted_text.endswith("I"):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("l"):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("TL"):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("UL"):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("DL"):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("PL"):
        extracted_text = extracted_text[:-1] + "!"
    # The only -EL ending that I know about is "PANEL"
    if extracted_text.endswith("EL") and not (extracted_text.endswith("PANEL") or extracted_text.endswith("DUEL")):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("'"):
        extracted_text = extracted_text[:-1] + "!"
    if extracted_text.endswith("?"):
        extracted_text = extracted_text[:-1] + "!"
    # Replace some things that are in-string
    extracted_text = extracted_text.replace("CO ", "GO ")
    extracted_text = extracted_text.replace("EL ZND", "E! 2ND")
    extracted_text = extracted_text.replace("DL ZND", "E! 2ND")
    extracted_text = extracted_text.replace("ZND", "2ND")
    extracted_text = extracted_text.replace("BABYSHT", "BABYSIT")

    extracted_text = extracted_text.upper()

    return extracted_text

def do_ocr_multiple(image, left_percent, right_percent, top_percent, bottom_percent):
    """
    This function runs OCR on a region of a screenshot.

    Args:
        image (numpy array): The input image.
        left_percent (float): The left boundary as a percentage of the image width.
        right_percent (float): The right boundary as a percentage of the image width.
        top_percent (float): The top boundary as a percentage of the image height.
        bottom_percent (float): The bottom boundary as a percentage of the image height.

    Returns:
        List of Strings: Output of the OCR, cleaned from known gotchas and bugs. (returned in all uppercase, YMMV if you don't)
    """
    result = do_ocr_shared(image, left_percent, right_percent, top_percent, bottom_percent)

    extracted_text = []
    
    # This is terrible, btw.
    # This project is not meant to be my pride, it's supposed to be a proof of concept for Twitch chat control.
    # There are some things that could be changed, but EasyOCR's results are just as bad, so...
    # Don't have your player names end in "k", k?
    # (I considered "0k" etc, but it's possible to have non 10-rounded numbers)
    # Right now, this simply filters out the things that end like a money/credit/etc line ends:
    for text in result:
        if not text[1].upper().endswith('K'):
            extracted_text.append(text[1].upper())
    # The results have bounding boxes that could be used to reasonably determine what's on the same line
    # (ie did a name split?) versus what's on another line on the same card (ie money/credit/etc).
    # If we're in this function *correctly* from "Choose your Opponent", that already means that there
    # are either three or four players. That is: there are two or three cards to read here. That should
    # technically mean that there are only two or three entries in extracted_text right now. This won't check
    # that here, though. Let the calling function determine that in case there's a reason to use this
    # function for other things.

    return extracted_text

# Generic reusable matching system
def sift_matcher(screenshot_np, gray_template, crop_region, threshold_val):
    """
    This function reduces a lot of code reuse in check functions.
    Use this to process the main check screenshot (ie the spins / buttons / prompts).
    Afterward, handle the return.

    Args:
        screenshot_np (numpy array): The input image.
        gray_template (numpy array): The template image already in grayscale.
        crop_region (tuple): (left_percent, right_percent, top_percent, bottom_percent) of screenshot_np region.
        threshold_val (float): If >= 1, treated as int for matches. If less, treated as percentage of matches.

    Returns:
        Boolean: False if no match / etc, True if there are more matches than the threshold.
    """

    # Convert the screenshot to a format OpenCV can work with
    screenshot_cropped = crop_image(screenshot_np, *crop_region)
    gray_screenshot = cv2.cvtColor(screenshot_cropped, cv2.COLOR_BGR2GRAY)

    # Applying the SIFT function
    kp_template, des_template = sift.detectAndCompute(gray_template, None)
    kp_screenshot, des_screenshot = sift.detectAndCompute(gray_screenshot, None)
    
    if des_template is not None and des_screenshot is not None:
        # Finding matches
        matches = bf.knnMatch(des_template, des_screenshot, k=2)
    
        # Applying the ratio test
        good_matches = []
        for match in matches:
            if len(match) < 2:
                #print(f"Not enough matches, skipping.")
                return False
            m, n = match
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        
        # Calculate the dynamic threshold based on the number of key points in the template
        num_template_points = len(kp_template)
        if threshold_val >= 1.0:
            threshold = int(threshold_val)
        else:
            threshold = int(threshold_val * num_template_points)

        if len(good_matches) >= threshold:
            return True
    return False

def capture_window_screenshot():
    window = pywinctl.getWindowsWithTitle(window_title)
    if not window:
        print("Window not found! No screenshot.")
        return None

    window = window[0]  # Get the first matching window
    bbox = (window.left, window.top, window.width, window.height)

    # Capture the screenshot using PyAutoGUI
    screenshot = pyautogui.screenshot(region=bbox)
    return screenshot

def _handle_response(resp: requests.Response) -> Dict[str, Any]:
    """Print HTTP error and return empty dictionary."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"HTTP {resp.status_code}: {resp.text}")
        return {}

    try:
        return resp.json()
    except ValueError as exc:
        print("Response is not valid JSON")
        return {}

def create_poll(
    title: str,
    duration_seconds: int,
    options: List[str],
    timeout: Optional[float] = None,
) -> str:
    """
    Create a new poll using MSC.

    title: Human‑readable poll title.
    duration_seconds: Length of the poll in seconds.
    options: List of option as strings.
    timeout: Optional request timeout in seconds.

    Returns
    -------
    poll_id: The poll ID as a string.
    """
    url = f"{msc_location}/createpoll"
    payload = {
        "channel_id": channel_id,
        "title": title,
        "duration": duration_seconds,
        "options": options,
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    data = _handle_response(resp)

    if "poll_id" not in data:
        print("ERROR: No poll_id present in the returned data!")
        return ""
    return data["poll_id"]

def get_poll(poll_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Retrieve the current state of a poll from MSC

    Returns the raw JSON object produced by the MSC `GET /getpoll` API call handler.
    The structure matches the Go `twitch.Poll` type, e.g.:

    ```json
    {
        "id": "12345",
        "title": "What do you think, chat?",
        "status": "ACTIVE",
        "choices": [
            {"title": "Yes", "votes": 3},
            {"title": "No", "votes": 5}
        ],
        ...
    }
    ```
    """
    url = f"{msc_location}/getpoll"
    params = {"channel_id": channel_id, "poll_id": poll_id}
    resp = requests.get(url, params=params, timeout=timeout)
    return _handle_response(resp)

def _determine_result(poll: Dict[str, Any]) -> str:
    """
    Decide the result. If it's a single highest result already, return it.
    Otherwise, randomly choose between the tie.
    """
    choices = poll.get("choices", [])
    if not choices:
        print("FAILURE: NO CHOICES RETURNED IN _determine_result()")
        return "" # I don't know what circumstances can trigger this. The PoC won't handle this, though.

    max_votes = max(choice.get("votes", 0) for choice in choices)
    winning = [c["title"] for c in choices if c.get("votes", 0) == max_votes]

    if len(winning) == 1:
        return winning[0]
    else:
        selected_winner = random.choice(winning)
        tied = "; ".join(winning)
        print(f"The top tie options (at {max_votes} votes) are: {tied}; Randomly selected winner: {selected_winner}")
        return selected_winner


def wait_for_poll(
    poll_id: str,
    poll_interval: float = 1.0,
    timeout: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Block until the poll finishes (status != "ACTIVE").

    Parameters
    ----------
    poll_interval: Seconds between successive `GET /getpoll` calls.
    timeout: Optional overall timeout; ``None`` means wait indefinitely.

    Returns
    -------
    (completed, result_string)
        * ``completed``: ``True`` if the poll reached a terminal state,
          ``False`` if the call timed‑out.
        * ``result_string``: Human‑readable result or ``""`` when not completed.
    """
    start = time.time()
    while True:
        poll = get_poll(poll_id)
        status = poll.get("status", "").upper()

        if status != "ACTIVE":
            # Poll is finished; find the winner.
            return True, _determine_result(poll)

        # Keep polling the poll status; let the timeout cause an exit if applicable.
        if timeout is not None and (time.time() - start) >= timeout:
            return False, ""
        time.sleep(poll_interval)


# This is the function that you want to use to poll chat from the logic functions below!
def poll_chat(
    title: str,
    options: List[str],
) -> str:
    """
    Poll chat with a given title and choices for the poll.
    Automatically blocks until the poll has returned.
    If polling is disabled in the user settings above (use_msc == False),
    this function automatically choses one and returns.

    Parameters
    ----------
    title: Poll title as a string
    options: List of strings for options of the poll. Max 5 or it will fail.
    """
    duration_seconds = 15 # 15 is the minimum that Twitch allows

    if len(options) > 5 or len(options) < 2:
        print(f"List of options must be between 2 and 5; this list is {len(options)}")

    if use_msc:
        poll_id = create_poll(title=title, duration_seconds=duration_seconds, options=options)
        if poll_id == "":
            print("No poll ID returned, leaving.")
            return ""
        result = wait_for_poll(poll_id=poll_id) #-> Tuple[bool, str]
    else:
        print("MSC is disabled; randomly deciding poll result...")
        decided_result = random.choice(options)
        result = (True, decided_result)

    if not result[0]:
        print("Poll timeout exceeded, leaving.")
        return ""
    if result[1] == "":
        print("Poll returned an empty string, leaving.")
        return ""
    print(f"Poll {title} result is: {result[1]}")
    return result[1]


### DETECT / CHECK / ACT ON FUNCTIONS ###


def detect_wait(screenshot, gray_template):
    screenshot_np = np.array(screenshot)
    match_result = sift_matcher(screenshot_np, gray_template, (40, 40, 85, 0), 10)

    if not match_result:
        return False

    print(f"WAIT DETECTED!")
    return True

def detect_spin(screenshot, gray_template_tuple):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template_tuple[1], (60, 0, 0, 80), 0.42)

    if not match_result:
        return False

    print(f"SPIN DETECTED! ({gray_template_tuple[0]})")
    print("Pressing space...")
    pyautogui.keyDown('space')
    time.sleep(0.5)
    pyautogui.keyUp('space')
    print("Released space...")
    return True

# For "OK", "Next", and other buttons
def detect_button(screenshot, gray_template_tuple, threshold):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template_tuple[1], (35, 35, 50, 0), threshold)

    if not match_result:
        return False

    print(f"BUTTON DETECTED ({gray_template_tuple[0]})")
    print("Pressing left...")
    pyautogui.press('left') # highlights the button
    print("Pressing enter...")
    pyautogui.press('enter')
    return True

# "Choose your Start in Life!"
def detect_cysil(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.5)

    if not match_result:
        return False

    print(f"CHOOSE YOUR START IN LIFE DETECTED! Options: Left = College; Right = Career")
    decision = poll_chat("Choose your Start in Life!", ["College", "Career"])
    print(f"Decision: {decision}!")
    if decision == "College":
        print("Pressing left for College...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Career":
        print("Pressing right for Career...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Which way to go?"
def detect_fork(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.5)

    if not match_result:
        return False

    # This can be improved. There are four different "Which way to go?" forks:
    # - Stay Single / Get Married
    # - Investment / Family
    # - Online Course / Continue Life
    # - Risky / Safe
    # (plus the "Choose your Start in Live!" at the beginning that is functionally the same)
    # The only difference between the non-start four are the things on the left and right cards.
    # That being said, for now, I'm staying only with a choice of left and right.

    print(f"FORK IN THE ROAD DETECTED! Left or right?")
    decision = poll_chat("Which way to go?", ["Left", "Right"])
    print(f"Decision: {decision}!")
    if decision == "Left":
        print("Pressing left...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Right":
        print("Pressing right...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Choose an Action!"
def detect_action(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.45)

    if not match_result:
        return False

    print(f"CHOOSE AN ACTION DETECTED! Reading options...")

    left_option = "Left Option"
    right_option = "Right Option"

    if use_ocr:
        left_option = do_ocr(screenshot_np, 25, 52, 28, 53)
        right_option = do_ocr(screenshot_np, 52, 25, 28, 53)

    print(f"Left option: {left_option}; Right option: {right_option}")
    decision = poll_chat("Which action card?",["Left", "Right"])
    print(f"Decision: {decision}!")
    if decision == "Left":
        print(f"Pressing left for {left_option}...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Right":
        print(f"Pressing right for {right_option}...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# Used for job choices: choose a job, improve job decision, etc.
def detect_job(screenshot, gray_template_tuple):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template_tuple[1], (59, 0, 0, 80), 0.4)

    if not match_result:
        return False

    print(f"JOB CHOICE DETECTED! ({gray_template_tuple[0]}) Reading options...")

    left_option = "Left Option"
    right_option = "Right Option"

    if use_ocr:
        left_option = do_ocr(screenshot_np, 25, 52, 28, 53)
        right_option = do_ocr(screenshot_np, 52, 25, 28, 53)

    print(f"Left option: {left_option}; Right option: {right_option}")
    decision = poll_chat("Which job card?",["Left", "Right"])
    print(f"Decision: {decision}!")
    if decision == "Left":
        print(f"Pressing left for {left_option}...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Right":
        print(f"Pressing right for {right_option}...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Choose your Spouse!"
def detect_spouse(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.5)

    if not match_result:
        return False

    # "Choose your Spouse!" can be confused with "Choose your House!"
    buy_house_card = is_card(screenshot_np, 40, 55, 40, 55)
    if buy_house_card:
        return False

    print(f"CHOOSE YOUR SPOUSE DETECTED! Three options: Man, Woman, Other")
    decision = poll_chat("Which Spouse?",["Man", "Woman", "Other"])
    print(f"Decision: {decision}!")
    if decision == "Man":
        print(f"Pressing left for Man...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    if decision == "Woman":
        print(f"Pressing left then right for Woman...")
        pyautogui.press('left')
        pyautogui.press('right')
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Other":
        print(f"Pressing right for Other...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Grow your Family!"
def detect_family(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.5)

    if not match_result:
        return False

    print(f"GROW YOUR FAMILY DETECTED! Baby or Pet?")
    decision = poll_chat("Grow your family with what?", ["Baby", "Pet"])
    print(f"Decision: {decision}!")
    if decision == "Baby":
        print(f"Pressing left for Baby...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Pet":
        print(f"Pressing right for Pet...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Where to Retire?"
def detect_retire(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.45)

    if not match_result:
        return False

    print(f"WHERE TO RETIRE DETECTED! Collect Pension or Bucket List?")
    decision = poll_chat("Where to Retire?",["Collect Pension", "Bucket List"])
    print(f"Decision: {decision}!")
    if decision == "Collect Pension":
        print(f"Pressing left for Collect Pension...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Bucket List":
        print(f"Pressing right for Bucket List...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Choose a Bucket List card!"
def detect_bucket_list(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.5)

    if not match_result:
        return False

    print(f"BUCKET LIST CARD DETECTED! Reading options...")

    left_option = "Left Option"
    right_option = "Right Option"

    if use_ocr:
        left_option = do_ocr(screenshot_np, 25, 52, 28, 53)
        right_option = do_ocr(screenshot_np, 52, 25, 28, 53)

    print(f"Left option: {left_option}; Right option: {right_option}")
    decision = poll_chat("Which bucket list card?", ["Left", "Right"])
    print(f"Decision: {decision}!")
    if decision == "Left":
        print(f"Pressing left for {left_option}...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Right":
        print(f"Pressing right for {right_option}...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Select your Opponent!"
def detect_opponent(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.4)

    if not match_result:
        return False

    print(f"SELECT YOUR OPPONENT DETECTED! Reading options...")

    results = do_ocr_multiple(screenshot_np, 42, 45, 30, 15) # it's off-center stuff, yes. don't look or you won't un-see it.

    # This OCR is really buggy, so we're dealing with it.
    # Unfortunately, this function depends on the OCR to get the count in the right ballpark.

    option1 = ""
    option2 = ""
    option3 = ""

    if len(results) < 3:
        # Assume 2 cards
        option1 = "Top Button"
        option2 = "Bottom Button"
        if len(results) == 2 and use_ocr:
            option1 = results[0]
            option2 = results[1]
    else:
        # Assume 3 cards
        option1 = "Top Button"
        option2 = "Middle Button"
        option3 = "Bottom Button"
        if len(results) == 3 and use_ocr:
            option1 = results[0]
            option2 = results[1]
            option3 = results[2]

    print(f"Option 1: {option1}; Option 2: {option2}, Option 3: {option3}")
    if option3 == "":
        decision = poll_chat("Which opponent?", ["Top", "Bottom"])
        print(f"Decision: {decision}!")
        if decision == "Top":
            print(f"Pressing up for {option1}...")
            pyautogui.press('up') # highlights the button
            print("Pressing enter...")
            pyautogui.press('enter')
        elif decision == "Bottom":
            print(f"Pressing down for {option2}...")
            pyautogui.press('down') # highlights the button
            print("Pressing enter...")
            pyautogui.press('enter')
        return True
    else:
        decision = poll_chat("Which opponent?", ["Top", "Middle", "Bottom"])
        print(f"Decision: {decision}!")
        if decision == "Top":
            print(f"Pressing up for {option1}...")
            pyautogui.press('up') # highlights the button
            print("Pressing enter...")
            pyautogui.press('enter')
        if decision == "Middle":
            print(f"Pressing up and then down for {option2}...")
            pyautogui.press('up')
            pyautogui.press('down') # highlights the button
            print("Pressing enter...")
            pyautogui.press('enter')
        elif decision == "Bottom":
            print(f"Pressing down for {option3}...")
            pyautogui.press('down') # highlights the button
            print("Pressing enter...")
            pyautogui.press('enter')
        return True

# "Choose your House!" (This is a buy or sell card, not specific houses)
def detect_buysell_house(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.45)

    if not match_result:
        return False

    # You haven't detected anything yet

    buy_card = is_card(screenshot_np, 40, 55, 40, 55) # This one matches good for buy
    sell_card = is_card(screenshot_np, 60, 35, 40, 55) # This one matches good for sell.

    if not buy_card and not sell_card:
        return False

    # OK, you've detected something
    print(f"BUY OR SELL HOUSE DETECTED!")

    if not sell_card:
        # Implicitly True buy_card
        print(f"Pressing left for Buy (Sell is not an option, no decision here)")
        pyautogui.press('left')
        print("Pressing enter...")
        pyautogui.press('enter')
        return True

    print(f"Left option: Buy; Right option: Sell")
    decision = poll_chat("Buy or Sell a house?", ["Buy", "Sell"])
    print(f"Decision: {decision}!")
    if decision == "Buy":
        print(f"Pressing left for Buy...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Sell":
        print(f"Pressing right for Sell...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Choose a House!" (This is a buy screen for a house)
def detect_buy_house(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.40)

    if not match_result:
        return False

    # OK, you've detected something
    print(f"BUY A HOUSE DETECTED! Buy the left property, the right property, or do nothing?")

    # NOTE: OCR could be used here. But the OCR has been so bad in tests, I'm not adding it here right now.

    decision = poll_chat("Choose your House!", ["Left", "Right", "Nothing"])
    print(f"Decision: {decision}!")
    if decision == "Left":
        print(f"Pressing left to buy left property...")
        pyautogui.press('left')
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Right":
        print(f"Pressing right to buy right property...")
        pyautogui.press('right')
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Nothing":
        print(f"Pressing down to do nothing...")
        pyautogui.press('down')
        print("Pressing enter...")
        pyautogui.press('enter')
        # There's an additional menu here that prompts about skipping turn. The check is on the right.
        time.sleep(1)
        print("Pressing right to highlight the confirmation check...")
        pyautogui.press('right')
        print("Pressing enter...")
        pyautogui.press('enter')
    return True

# "Choose a House to Sell!" (This comes up if you choose to sell a house before the end)
# This is a bit of a gamble because I'm not sure how it works with more than one house.
# I haven't been able to buy two houses and sell before the end and I didn't see it online, either.
# Therefore, I'm assuming that the screen will half and have two house cards -- kind of like the buy
# screen -- if there are two houses. I'm checking if the single card is present in the middle.
# If it is, it's automatic. If not, it's a prompt. If there's more than two cards, this doesn't
# cover that scenario and I have no idea how to handle that correctly.
def detect_sell_house(screenshot, gray_template):
    screenshot_np = np.array(screenshot)

    match_result = sift_matcher(screenshot_np, gray_template, (60, 0, 0, 80), 0.5)

    if not match_result:
        return False

    print(f"CHOOSE A HOUSE TO SELL DETECTED!")

    single_card = is_card(screenshot_np, 48, 48, 77, 20)

    if single_card:
        # Implicitly True buy_card
        print(f"Pressing left to highlight single option.")
        pyautogui.press('left')
        print("Pressing enter...")
        pyautogui.press('enter')
        return True

    decision = poll_chat("Which house to sell?", ["Left", "Right"])
    print(f"Decision: {decision}!")
    if decision == "Left":
        print(f"Pressing left...")
        pyautogui.press('left') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    elif decision == "Right":
        print(f"Pressing right...")
        pyautogui.press('right') # highlights the button
        print("Pressing enter...")
        pyautogui.press('enter')
    return True



## TODO: House sell offers here ##


### MAIN LOOP ###

while 1:
    time.sleep(2)
    if cursor_short():
        # Short out the automatic control, the operator is doing something.
        continue
    screenshot = capture_window_screenshot()
    continueloop = False # Used after for-loops since Python doesn't support loop labelling
    if screenshot:
        wait_check = detect_wait(screenshot, gray_wait)
        if wait_check:
            time.sleep(2) # Wait a little longer
            continue
        for gray_button in gray_buttons:
            button_check = detect_button(screenshot, gray_button, 0.35)
            if button_check:
                time.sleep(4) # Wait a little longer
                continueloop = True
                break
        if continueloop:
            continue
        for gray_spin in gray_spins:
            spin_check = detect_spin(screenshot, gray_spin)
            if spin_check:
                time.sleep(4) # Wait a little longer
                continueloop = True
                break
        if continueloop:
            continue
        cysil_check = detect_cysil(screenshot, gray_choose_your_start_in_life)
        if cysil_check:
            # don't wait extra here, we have to spin right after
            continue
        fork_check = detect_fork(screenshot, gray_which_way_to_go)
        if fork_check:
            # don't wait extra here, we have to spin right after
            continue
        action_check = detect_action(screenshot, gray_choose_an_action)
        if action_check:
            time.sleep(2) # Wait a little longer
            continue
        for gray_job in gray_jobs:
            job_check = detect_job(screenshot, gray_job)
            if job_check:
                time.sleep(2) # Wait a little longer
                continueloop = True
                break
        if continueloop:
            continue
        spouse_check = detect_spouse(screenshot, gray_choose_your_spouse)
        if spouse_check:
            time.sleep(2) # Wait a little longer. It's a cutscene and then a spin starts.
            continue
        family_check = detect_family(screenshot, gray_grow_your_family)
        if family_check:
            time.sleep(2) # Wait a little longer. It's a cutscene after.
            continue
        retire_check = detect_retire(screenshot, gray_where_to_retire)
        if retire_check:
            time.sleep(2) # Wait a little longer. It's a cutscene after.
            continue
        bucket_list_check = detect_bucket_list(screenshot, gray_choose_a_bucket_list_card)
        if bucket_list_check:
            # No need to wait
            continue
        opponent_check = detect_opponent(screenshot, gray_select_your_opponent)
        if opponent_check:
            # A spin comes after this
            continue
        buysell_house_check = detect_buysell_house(screenshot, gray_choose_your_house_buysell)
        if buysell_house_check:
            # More selections come after this.
            continue
        buy_house_check = detect_buy_house(screenshot, gray_choose_a_house_buy)
        if buy_house_check:
            time.sleep(2) # Cutscene and end of move
            continue
        sell_house_check = detect_sell_house(screenshot, gray_choose_a_house_sell)
        if sell_house_check:
            # No extra sleep, need to choose offers
            continue


        # This should be the last check.
        # It's a re-check for the buttons that need more sensitivity.
        # This is at the end to avoid false positives from the low threshold value.
        for gray_button in gray_money_buttons:
            button_check = detect_button(screenshot, gray_button, 0.28)
            if button_check:
                time.sleep(4) # Wait a little longer
                continueloop = True
                break
        if continueloop:
            continue

        print("Nothing was detected...")

