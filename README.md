# Chat Plays Game of Life 2 via Twitch Polls! (Proof of Concept)

This proof of concept enables Twitch Chat to use Twitch polls to control a character in [Game of Life 2](https://store.steampowered.com/app/1455630/THE_GAME_OF_LIFE_2/)!

Technologies used in this proof of concept include OpenCV (the opencv-python library), OCR (the EasyOCR library), a Python HID controller (the PyAutoGUI and PyWinCtl libraries), and [Monktype's Stream Commands (MSC)](https://github.com/Monktype/msc) to act as a reusable API target while interacting with the Twitch API.

The tool in this repository is not really intended to be *used*, rather it's intended to be pointed at as an example of Twitch Chat interactivity. (If you use it, beware of known issues and unknown issues.)
In general, Twitch Chat interactivity has three different first-party possibilities that do not use channel points or Twitch bits: Twitch extensions, Twitch polls, and Twitch chat directly:
- Twitch extensions, like that used in Jerma's Dollhouse for Twitch Chat interactivity, has a benefit in that it can be integrated into the video player. However, it has a drawback of being more complex, requiring more development and extra infrastructure to run. It also doesn't necessarily align with player latency -- any extension controls may appear out-of-sync with the stream's video.
- Twitch polls may be triggered from the Twitch API directly. Inherent benefits of using Twitch polls to receive Chat input are that there is neither latency issues between a Chat member seeing the option and voting nor is there a potential vote loss (outside of Twitch breaking). The Chat may clearly see what is being voted on by a title and written options which are clicked directly. The polling time is controlled centrally by Twitch, where only one poll may be run at a time on a channel, which are clearly separated by UUIDs on the backend. Downsides of Twitch polls include the current maximum of only five voting options at once, the requirement to interface with the Twitch API (or mod/streamer controls directly), and that third-party interfaces, like Chatterino and DankChat, etc, are unable to interface with Twitch polls. It is necessary for a channel to be Twitch Affiliate or Twitch Partner to run polls. There is also no (API-controllable) equivalent to Twitch polls on YouTube and Kick at this time.
- Control by the Twitch chat interface directly has been used in many circumstances already. It's simple to have an unauthenticated tool connect to an IRC room and parse messages. But Twitch Chat is reacting to the stream over a range of *several seconds* due to the video player's latency and viewers' reaction/typing times. This can be further affected in channels where slowmode is enabled, where some viewers may send a command from several seconds before, trying to "slip in" their message. Depending on the implementation of the Twitch chat interface controller or the scenario where it's being used, the controller may be registering two or more conflicting "polled" actions from the Chat at once even though the Chat is working towards the the same goal. To see an example of this, watch a Chat-controlled Pokemon stream with a large audience -- conflicting Chat messages happens regularly. Also, it's possible to see that either Twitch's chat backend or controllers' chat connections to the Twitch backend drop some messages in busy chatrooms, which can further frustrate chatters and degrade the quality of the interactivity.

Game of Life 2 was chosen as a proof of concept target because its controls are simple enough to fit into the poll requirements: no more than five clear options at a time (in most cases) with enough time provided by the game for the CV/OCR to execute, a poll to run, and the HID controller to make a selection. The intended focus of this proof of concept is the poll-based controls, not the specific game and CV/OCR bugs that come with controlling the specific game.

A game controlled by Twitch polls is not completely novel to this repository: [RamenBucket created a chess controller](https://github.com/RamenBucket/twitch-chess), for example; there may be better-known examples.


## Setup

### MSC Setup

1. Download or build MSC from [its GitHub page](https://github.com/Monktype/msc).
2. Get an API key from Twitch. (Refer to MSC's page.)
3. Configure API key in MSC / authenticate. (Refer to MSC's page.)
4. Get your target's channel numerical ID if you don't already have that (`msc userid <channel>`).

### Python and Script Setup

1. Setup a venv (or don't, you do you).
2. Install Python libraries (pywinctl, pyautogui, opencv-python, EasyOCR, numpy, requests).
3. In the top of the script, there are four user-defined settings: `use_ocr`, `msc_location`, `use_msc` (if False, choices will be entirely random), and `channel_id` (the target channel's numerical ID as a string). Change these as necessary.

## Run

1. Start MSC's local API (`msc api`), defaults to port `8080`.
2. Run the script (`python3 game-of-life-controller.py`). EasyOCR will download model data on the first run.
3. Start the game. Note that your mouse being anywhere in the game window except the upper left corner will stop the script from processing the window. If your mouse cursor is off the game window or in the upper left of the game window, the controller will process the window.

## Proof of Concept Limitations, Issues

- This proof of concept is missing house selling mechanics.
- OpenCV is only doing feature matching based on template pictures. There are false positives and false negatives.
- EasyOCR can return junk when used against this game's text. Beware if you set `use_ocr = True`.
- Development and testing was only done in 1280x768 windowed mode on an Ubuntu virtual machine.
- "It should work" on Windows with other resolutions (the library choice was based on cross-platform support), but it hasn't been tested. PRs welcomed.
- This proof of concept is only intended to illustrate Twitch Chat control via Twitch polls in an existing game while illustrating some current limitations and flows of Twitch polls.
- All human characters with a frontend in the controlled game window (ie local human players) will be controlled by this tool. For human multiplayer with this controller tool, use networked multiplayer (or move your mouse into the game window to stop the tool from reading the window).

