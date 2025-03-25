#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on March 24, 2025, at 01:27
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_2
from pylsl import StreamInfo, StreamOutlet # import required classes
info = StreamInfo(name='Trigger', type='Markers', channel_count=1,
channel_format='int32', source_id='Example') # sets variables for object info
outlet = StreamOutlet(info) # initialize stream.
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'EmotionVR_task'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'Age': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1920, 1200]
_loggingLevel = logging.getLevel('data')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\devsc\\OneDrive - Ontario Tech University\\fNIRS_Emotions\\EmotionVR_task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('', )
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('InstructionResp') is None:
        # initialise InstructionResp
        InstructionResp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='InstructionResp',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('breakResp') is None:
        # initialise breakResp
        breakResp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='breakResp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instruction_routine" ---
    instructionText = visual.TextStim(win=win, name='instructionText',
        text='Welcome to the experiment!\n\nHere are the instructions:\n\n1. In phase 1, you will be presented with 8 faces in rapid succession, pay very careful attention to these images. \n\n2. In phase 2, you will be shown a single face. Your task is to determine if you saw that face in phase 1 (first 8 faces). \n\n3. Press y for Yes, or n for No\n\nWhen you are ready: Press SPACEBAR to continue\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    InstructionResp = keyboard.Keyboard(deviceName='InstructionResp')
    
    # --- Initialize components for Routine "fixation_routine" ---
    fixationCross = visual.ShapeStim(
        win=win, name='fixationCross', vertices='cross',
        size=[0.1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from code_5
    break_counter = 0
    
    # --- Initialize components for Routine "trial_routine" ---
    # Run 'Begin Experiment' code from code
    iti = 0
    trialImage = visual.ImageStim(
        win=win,
        name='trialImage', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[1, 1],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    jitter = visual.TextStim(win=win, name='jitter',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "task_routine" ---
    taskImage = visual.ImageStim(
        win=win,
        name='taskImage', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[1, 1],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    taskText = visual.TextStim(win=win, name='taskText',
        text='Did you see this image? Press y for Yes, n for No',
        font='Open Sans',
        pos=(0, 0.475), height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "break_routine" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='Take a break!\n\nWhen you are ready: Press SPACEBAR to continue\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=1.75, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    breakResp = keyboard.Keyboard(deviceName='breakResp')
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Open Sans',
        pos=(0, -0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prog = visual.Progress(
        win, name='prog',
        progress=0.0,
        pos=(-0.5, -0.5), size=(1, 0.1), anchor='bottom-left', units='height',
        barColor='white', backColor=None, borderColor='white', colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-4
    )
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instruction_routine" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_routine.started', globalClock.getTime(format='float'))
    # create starting attributes for InstructionResp
    InstructionResp.keys = []
    InstructionResp.rt = []
    _InstructionResp_allKeys = []
    # keep track of which components have finished
    Instruction_routineComponents = [instructionText, InstructionResp]
    for thisComponent in Instruction_routineComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction_routine" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructionText* updates
        
        # if instructionText is starting this frame...
        if instructionText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructionText.frameNStart = frameN  # exact frame index
            instructionText.tStart = t  # local t and not account for scr refresh
            instructionText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructionText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructionText.started')
            # update status
            instructionText.status = STARTED
            instructionText.setAutoDraw(True)
        
        # if instructionText is active this frame...
        if instructionText.status == STARTED:
            # update params
            pass
        
        # *InstructionResp* updates
        waitOnFlip = False
        
        # if InstructionResp is starting this frame...
        if InstructionResp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            InstructionResp.frameNStart = frameN  # exact frame index
            InstructionResp.tStart = t  # local t and not account for scr refresh
            InstructionResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(InstructionResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'InstructionResp.started')
            # update status
            InstructionResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(InstructionResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(InstructionResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if InstructionResp.status == STARTED and not waitOnFlip:
            theseKeys = InstructionResp.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _InstructionResp_allKeys.extend(theseKeys)
            if len(_InstructionResp_allKeys):
                InstructionResp.keys = _InstructionResp_allKeys[-1].name  # just the last key pressed
                InstructionResp.rt = _InstructionResp_allKeys[-1].rt
                InstructionResp.duration = _InstructionResp_allKeys[-1].duration
                # was this correct?
                if (InstructionResp.keys == str('')) or (InstructionResp.keys == ''):
                    InstructionResp.corr = 1
                else:
                    InstructionResp.corr = 0
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction_routineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction_routine" ---
    for thisComponent in Instruction_routineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruction_routine.stopped', globalClock.getTime(format='float'))
    # check responses
    if InstructionResp.keys in ['', [], None]:  # No response was made
        InstructionResp.keys = None
        # was no response the correct answer?!
        if str('').lower() == 'none':
           InstructionResp.corr = 1;  # correct non-response
        else:
           InstructionResp.corr = 0;  # failed to respond (incorrectly)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('InstructionResp.keys',InstructionResp.keys)
    thisExp.addData('InstructionResp.corr', InstructionResp.corr)
    if InstructionResp.keys != None:  # we had a response
        thisExp.addData('InstructionResp.rt', InstructionResp.rt)
        thisExp.addData('InstructionResp.duration', InstructionResp.duration)
    thisExp.nextEntry()
    # the Routine "Instruction_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks_stim = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('blocks.xlsx'),
        seed=None, name='blocks_stim')
    thisExp.addLoop(blocks_stim)  # add the loop to the experiment
    thisBlocks_stim = blocks_stim.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_stim.rgb)
    if thisBlocks_stim != None:
        for paramName in thisBlocks_stim:
            globals()[paramName] = thisBlocks_stim[paramName]
    
    for thisBlocks_stim in blocks_stim:
        currentLoop = blocks_stim
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlocks_stim.rgb)
        if thisBlocks_stim != None:
            for paramName in thisBlocks_stim:
                globals()[paramName] = thisBlocks_stim[paramName]
        
        # --- Prepare to start Routine "fixation_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation_routine.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_5
        outlet.push_sample(x=[blocknumber])
        break_counter += 1
        
        
        # keep track of which components have finished
        fixation_routineComponents = [fixationCross]
        for thisComponent in fixation_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 16.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixationCross* updates
            
            # if fixationCross is starting this frame...
            if fixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationCross.frameNStart = frameN  # exact frame index
                fixationCross.tStart = t  # local t and not account for scr refresh
                fixationCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationCross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationCross.started')
                # update status
                fixationCross.status = STARTED
                fixationCross.setAutoDraw(True)
            
            # if fixationCross is active this frame...
            if fixationCross.status == STARTED:
                # update params
                pass
            
            # if fixationCross is stopping this frame...
            if fixationCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationCross.tStartRefresh + 16-frameTolerance:
                    # keep track of stop time/frame for later
                    fixationCross.tStop = t  # not accounting for scr refresh
                    fixationCross.tStopRefresh = tThisFlipGlobal  # on global time
                    fixationCross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationCross.stopped')
                    # update status
                    fixationCross.status = FINISHED
                    fixationCross.setAutoDraw(False)
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_routine" ---
        for thisComponent in fixation_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation_routine.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-16.000000)
        
        # set up handler to look after randomisation of conditions etc
        trials_stim = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(blocknames),
            seed=None, name='trials_stim')
        thisExp.addLoop(trials_stim)  # add the loop to the experiment
        thisTrials_stim = trials_stim.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_stim.rgb)
        if thisTrials_stim != None:
            for paramName in thisTrials_stim:
                globals()[paramName] = thisTrials_stim[paramName]
        
        for thisTrials_stim in trials_stim:
            currentLoop = trials_stim
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_stim.rgb)
            if thisTrials_stim != None:
                for paramName in thisTrials_stim:
                    globals()[paramName] = thisTrials_stim[paramName]
            
            # --- Prepare to start Routine "trial_routine" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial_routine.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code
            iti = randint(5,15)*0.05
            trialImage.setImage(filepaths)
            jitter.setText('')
            # Run 'Begin Routine' code from code_3
            outlet.push_sample(x=[marker])
            # keep track of which components have finished
            trial_routineComponents = [trialImage, jitter]
            for thisComponent in trial_routineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_routine" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trialImage* updates
                
                # if trialImage is starting this frame...
                if trialImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trialImage.frameNStart = frameN  # exact frame index
                    trialImage.tStart = t  # local t and not account for scr refresh
                    trialImage.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trialImage, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trialImage.started')
                    # update status
                    trialImage.status = STARTED
                    trialImage.setAutoDraw(True)
                
                # if trialImage is active this frame...
                if trialImage.status == STARTED:
                    # update params
                    pass
                
                # if trialImage is stopping this frame...
                if trialImage.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trialImage.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        trialImage.tStop = t  # not accounting for scr refresh
                        trialImage.tStopRefresh = tThisFlipGlobal  # on global time
                        trialImage.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trialImage.stopped')
                        # update status
                        trialImage.status = FINISHED
                        trialImage.setAutoDraw(False)
                
                # *jitter* updates
                
                # if jitter is starting this frame...
                if jitter.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                    # keep track of start time/frame for later
                    jitter.frameNStart = frameN  # exact frame index
                    jitter.tStart = t  # local t and not account for scr refresh
                    jitter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(jitter, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'jitter.started')
                    # update status
                    jitter.status = STARTED
                    jitter.setAutoDraw(True)
                
                # if jitter is active this frame...
                if jitter.status == STARTED:
                    # update params
                    pass
                
                # if jitter is stopping this frame...
                if jitter.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > jitter.tStartRefresh + iti-frameTolerance:
                        # keep track of stop time/frame for later
                        jitter.tStop = t  # not accounting for scr refresh
                        jitter.tStopRefresh = tThisFlipGlobal  # on global time
                        jitter.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'jitter.stopped')
                        # update status
                        jitter.status = FINISHED
                        jitter.setAutoDraw(False)
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_routineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_routine" ---
            for thisComponent in trial_routineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial_routine.stopped', globalClock.getTime(format='float'))
            # the Routine "trial_routine" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_stim'
        
        
        # --- Prepare to start Routine "task_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('task_routine.started', globalClock.getTime(format='float'))
        taskImage.setImage(tasknames)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from code_4
        outlet.push_sample(x=[taskmarker])
        # keep track of which components have finished
        task_routineComponents = [taskImage, taskText, key_resp]
        for thisComponent in task_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *taskImage* updates
            
            # if taskImage is starting this frame...
            if taskImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                taskImage.frameNStart = frameN  # exact frame index
                taskImage.tStart = t  # local t and not account for scr refresh
                taskImage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(taskImage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'taskImage.started')
                # update status
                taskImage.status = STARTED
                taskImage.setAutoDraw(True)
            
            # if taskImage is active this frame...
            if taskImage.status == STARTED:
                # update params
                pass
            
            # if taskImage is stopping this frame...
            if taskImage.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > taskImage.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    taskImage.tStop = t  # not accounting for scr refresh
                    taskImage.tStopRefresh = tThisFlipGlobal  # on global time
                    taskImage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'taskImage.stopped')
                    # update status
                    taskImage.status = FINISHED
                    taskImage.setAutoDraw(False)
            
            # *taskText* updates
            
            # if taskText is starting this frame...
            if taskText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                taskText.frameNStart = frameN  # exact frame index
                taskText.tStart = t  # local t and not account for scr refresh
                taskText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(taskText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'taskText.started')
                # update status
                taskText.status = STARTED
                taskText.setAutoDraw(True)
            
            # if taskText is active this frame...
            if taskText.status == STARTED:
                # update params
                pass
            
            # if taskText is stopping this frame...
            if taskText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > taskText.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    taskText.tStop = t  # not accounting for scr refresh
                    taskText.tStopRefresh = tThisFlipGlobal  # on global time
                    taskText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'taskText.stopped')
                    # update status
                    taskText.status = FINISHED
                    taskText.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['y','n'], ignoreKeys=None, waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_routine" ---
        for thisComponent in task_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('task_routine.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        blocks_stim.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            blocks_stim.addData('key_resp.rt', key_resp.rt)
            blocks_stim.addData('key_resp.duration', key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "break_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_routine.started', globalClock.getTime(format='float'))
        # create starting attributes for breakResp
        breakResp.keys = []
        breakResp.rt = []
        _breakResp_allKeys = []
        text.setText(str(break_counter) + " blocks out of 56 done!")
        # Run 'Begin Routine' code from code_6
        if break_counter % 7 != 0 or break_counter == 56:
            continueRoutine = False
            
        prog.setProgress(break_counter / 56)
        # keep track of which components have finished
        break_routineComponents = [break_text, breakResp, text, prog]
        for thisComponent in break_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # *breakResp* updates
            waitOnFlip = False
            
            # if breakResp is starting this frame...
            if breakResp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                breakResp.frameNStart = frameN  # exact frame index
                breakResp.tStart = t  # local t and not account for scr refresh
                breakResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakResp.started')
                # update status
                breakResp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(breakResp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(breakResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if breakResp.status == STARTED and not waitOnFlip:
                theseKeys = breakResp.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
                _breakResp_allKeys.extend(theseKeys)
                if len(_breakResp_allKeys):
                    breakResp.keys = _breakResp_allKeys[-1].name  # just the last key pressed
                    breakResp.rt = _breakResp_allKeys[-1].rt
                    breakResp.duration = _breakResp_allKeys[-1].duration
                    # was this correct?
                    if (breakResp.keys == str('')) or (breakResp.keys == ''):
                        breakResp.corr = 1
                    else:
                        breakResp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *prog* updates
            
            # if prog is starting this frame...
            if prog.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog.frameNStart = frameN  # exact frame index
                prog.tStart = t  # local t and not account for scr refresh
                prog.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog.started')
                # update status
                prog.status = STARTED
                prog.setAutoDraw(True)
            
            # if prog is active this frame...
            if prog.status == STARTED:
                # update params
                pass
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_routine" ---
        for thisComponent in break_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_routine.stopped', globalClock.getTime(format='float'))
        # check responses
        if breakResp.keys in ['', [], None]:  # No response was made
            breakResp.keys = None
            # was no response the correct answer?!
            if str('').lower() == 'none':
               breakResp.corr = 1;  # correct non-response
            else:
               breakResp.corr = 0;  # failed to respond (incorrectly)
        # store data for blocks_stim (TrialHandler)
        blocks_stim.addData('breakResp.keys',breakResp.keys)
        blocks_stim.addData('breakResp.corr', breakResp.corr)
        if breakResp.keys != None:  # we had a response
            blocks_stim.addData('breakResp.rt', breakResp.rt)
            blocks_stim.addData('breakResp.duration', breakResp.duration)
        # the Routine "break_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'blocks_stim'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
