'use strict';
import { FoodType } from "./FoodType.js";
import { BarbarianAssault } from "./BarbarianAssault.js";
import { Renderer } from "./Renderer.js";
import { Position } from "./Position.js";
import { LOS_EAST_MASK, LOS_FULL_MASK, LOS_NORTH_MASK, LOS_SOUTH_MASK, LOS_WEST_MASK, MOVE_EAST_MASK, MOVE_FULL_MASK, MOVE_NORTH_MASK, MOVE_SOUTH_MASK, MOVE_WEST_MASK } from "./BarbarianAssaultMap.js";
import { RunnerPenance } from "./RunnerPenance.js";
import { MoveCommand } from "./MoveCommand.js";
import { DefenderActionCommand } from "./DefenderActionCommand.js";
import { DefenderActionType } from "./DefenderActionType.js";
import { TileMarker } from "./TileMarker.js";
import { RGBColor } from "./RGBColor.js";
import { Player } from "./Player.js";
import { HealerTargetType } from "./HealerTargetType.js";
import { parseCannonInput, getCannonPosition } from "./Cannon.js";
import { CannonSide } from "./CannonSide.js";
import { parseHealerCodes, assignSpawnPriorities } from "./HealerCodeAction.js";
const HTML_CANVAS = "basimcanvas";
const HTML_RUNNER_MOVEMENTS = "runnermovements";
const HTML_START_BUTTON = "wavestart";
const HTML_WAVE_SELECT = "waveselect";
const HTML_TICK_COUNT = "tickcount";
const HTML_DEF_LEVEL_SELECT = "deflevelselect";
const HTML_TOGGLE_REPAIR = 'togglerepair';
const HTML_TOGGLE_PAUSE_SL = 'togglepausesl';
const HTML_CURRENT_DEF_FOOD = "currdeffood";
const HTML_TICK_DURATION = "tickduration";
const HTML_TOGGLE_INFINITE_FOOD = "toggleinfinitefood";
const HTML_TOGGLE_LOG_TO_REPAIR = "togglelogtorepair";
const HTML_MARKER_COLOR = "marker";
const HTML_MARKING_TILES = "markingtiles";
const HTML_MAIN_ATTACKER_COMMANDS = "mainattackercommands";
const HTML_SECOND_ATTACKER_COMMANDS = "secondattackercommands";
const HTML_HEALER_COMMANDS = "healercommands";
const HTML_COLLECTOR_COMMANDS = "collectorcommands";
const HTML_DEFENDER_COMMANDS = "defendercommands";
const HTML_PLAYER_SELECT = "playerselect";
const HTML_CONTROLLED_COMMANDS = "controlledcommands";
const HTML_FOOD_CALLS = "foodcalls";
const HTML_RUNNER_MOVEMENTS_TO_CHECK = "runnermovementstocheck";
const HTML_RUNNERS_DEAD_BY_TICK = "runnersdeadbytick";
const HTML_SIMULATE = "simulate";
const HTML_RUNNERS_DO_NOT_DIE_WITH_MOVEMENTS = "runnersdonotdiemovements";
const HTML_CANNON_QUEUE = "cannonqueue";
const HTML_RUNNER_TABLE = "runnertable";
const HTML_HEALER_TABLE = "healertable";
const HTML_TOGGLE_DARK_MODE = "toggledarkmode";
const HTML_TOGGLE_RENDER_DISTANCE = "togglerenderdistance";
const HTML_TOGGLE_SIMPLE_FOOD = "togglesimplefood";
const HTML_FOOD_CALLS_ROW = "foodcallsrow";
const HTML_INFINITE_FOOD_ROW = "infinitefoodrow";
const HTML_CURRENT_FOOD_ROW = "currfoodrow";
const HTML_HEALER_CODES = "healercodes";
const HTML_HEALER_SPAWN_TARGETS = "healerspawntargets";
const HTML_RUNNER_SPAWNS = "runnerspawns";
const HTML_HEALER_SPAWNS = "healerspawns";
const HTML_COPY_CONTROLLED_COMMANDS = "copycontrolledcommands";
const HTML_EXPORT_MARKERS = "exportmarkers";
const HTML_IMPORT_MARKERS = "importmarkers";
const HTML_MARKER_IMPORT_FIELD = "markerimportfield";
const HTML_SETTINGS_EXPORT = "settingsexport";
const HTML_SETTINGS_IMPORT = "settingsimport";
const HTML_SETTINGS_IMPORT_FIELD = "settingsimportfield";
const HTML_PAUSE_RESUME = "pauseresume";
const HTML_STEP_BACK = "stepback";
const HTML_STEP_FORWARD = "stepforward";
window.onload = init;
var markingTiles;
var markedTiles;
var canvas;
var movementsInput;
var tickDurationInput;
var startStopButton;
var waveSelect;
var defenderLevelSelection;
var toggleRepair;
var togglePauseSaveLoad;
var toggleInfiniteFood;
var toggleLogToRepair;
var tickCountSpan;
var currentDefenderFoodSpan;
var markerColorInput;
var isRunning = false;
var barbarianAssault;
var infiniteFood;
var simpleFood = true;
var isPaused;
var pauseSaveLoad;
var saveExists;
var renderer;
var requireLogs;
var requireRepairs;
var tickTimerId;
var wave;
var defenderLevel;
var markerColor;
var toggleRenderDistance;
var toggleMarkingTiles;
var playerSelect;
var player;
var controlledCommands;
var foodCallsInput;
var runnerMovementsToCheckInput;
var runnersDeadByTickInput;
var simulateButton;
var runnersDoNotDieWithMovements;
var cannonQueueInput;
var healerCodesInput;
var healerSpawnTargetsInput;
var runnerSpawnsInput;
var healerSpawnsInput;
var pauseResumeButton;
var stepBackButton;
var stepForwardButton;
const STATE_HISTORY_LIMIT = 1000;
var stateHistory = [];
var stateIndex = -1;
var savedBarbarianAssault;
var savedTickCountSpanInnerHTML;
var savedCurrentDefenderFoodSpanInnerHTML;
var savedPlayer;
var savedControlledCommandsInnerHTML;
var savedDefenderLevel;
var savedWave;
var savedMovementsString;
var savedMainAttackerCommands;
var savedSecondAttackerCommands;
var savedHealerCommands;
var savedCollectorCommands;
var savedDefenderCommands;
var savedRequireRepairs;
var savedInfiniteFood;
var savedRequireLogs;
var savedFoodCallsString;
var savedCannonQueueString;
var savedHealerCodesString;
var savedHealerSpawnTargetsString;
var savedRunnerSpawnsString;
var savedHealerSpawnsString;
/**
 * Initializes the simulator.
 */
function init() {
    canvas = document.getElementById(HTML_CANVAS);
    movementsInput = document.getElementById(HTML_RUNNER_MOVEMENTS);
    movementsInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    movementsInput.onchange = movementsInputOnChange;
    foodCallsInput = document.getElementById(HTML_FOOD_CALLS);
    foodCallsInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    foodCallsInput.onchange = foodCallsInputOnChange;
    tickDurationInput = document.getElementById(HTML_TICK_DURATION);
    startStopButton = document.getElementById(HTML_START_BUTTON);
    startStopButton.onclick = startStopButtonOnClick;
    waveSelect = document.getElementById(HTML_WAVE_SELECT);
    waveSelect.onchange = waveSelectOnChange;
    defenderLevelSelection = document.getElementById(HTML_DEF_LEVEL_SELECT);
    defenderLevelSelection.onchange = defenderLevelSelectionOnChange;
    toggleRepair = document.getElementById(HTML_TOGGLE_REPAIR);
    toggleRepair.onchange = toggleRepairOnChange;
    togglePauseSaveLoad = document.getElementById(HTML_TOGGLE_PAUSE_SL);
    togglePauseSaveLoad.onchange = togglePauseSaveLoadOnChange;
    toggleInfiniteFood = document.getElementById(HTML_TOGGLE_INFINITE_FOOD);
    toggleInfiniteFood.onchange = toggleInfiniteFoodOnChange;
    toggleLogToRepair = document.getElementById(HTML_TOGGLE_LOG_TO_REPAIR);
    toggleLogToRepair.onchange = toggleLogToRepairOnChange;
    toggleRenderDistance = document.getElementById(HTML_TOGGLE_RENDER_DISTANCE);
    tickCountSpan = document.getElementById(HTML_TICK_COUNT);
    currentDefenderFoodSpan = document.getElementById(HTML_CURRENT_DEF_FOOD);
    markerColorInput = document.getElementById(HTML_MARKER_COLOR);
    renderer = new Renderer(canvas, 64 * 12, 48 * 12, 12);
    toggleMarkingTiles = document.getElementById(HTML_MARKING_TILES);
    toggleMarkingTiles.onchange = toggleMarkingTilesOnChange;
    markingTiles = toggleMarkingTiles.checked;
    markedTiles = [];
    infiniteFood = toggleInfiniteFood.checked;
    const toggleSimpleFoodButton = document.getElementById(HTML_TOGGLE_SIMPLE_FOOD);
    toggleSimpleFoodButton.onclick = function () {
        simpleFood = !simpleFood;
        toggleSimpleFoodButton.innerHTML = simpleFood ? "Disable simple food" : "Enable simple food";
        const display = simpleFood ? "none" : "";
        document.getElementById(HTML_FOOD_CALLS_ROW).style.display = display;
        document.getElementById(HTML_INFINITE_FOOD_ROW).style.display = display;
        document.getElementById(HTML_CURRENT_FOOD_ROW).style.display = display;
        document.getElementById("hotkeylegend-normal").style.display = display;
        document.getElementById("hotkeylegend-simple").style.display = simpleFood ? "" : "none";
    };
    requireRepairs = toggleRepair.checked;
    requireLogs = toggleLogToRepair.checked;
    pauseSaveLoad = togglePauseSaveLoad.checked;
    pauseResumeButton = document.getElementById(HTML_PAUSE_RESUME);
    stepBackButton = document.getElementById(HTML_STEP_BACK);
    stepForwardButton = document.getElementById(HTML_STEP_FORWARD);
    reset();
    window.onkeydown = windowOnKeyDown;
    canvas.onmousedown = canvasOnMouseDown;
    canvas.oncontextmenu = function (mouseEvent) {
        mouseEvent.preventDefault();
    };
    wave = Number(waveSelect.value);
    defenderLevel = Number(defenderLevelSelection.value);
    markerColor = Number("0x" + markerColorInput.value.substring(1));
    markerColorInput.onchange = markerColorInputOnChange;
    playerSelect = document.getElementById(HTML_PLAYER_SELECT);
    playerSelect.onchange = playerSelectOnChange;
    player = playerSelect.value;
    controlledCommands = document.getElementById(HTML_CONTROLLED_COMMANDS);
    runnerMovementsToCheckInput = document.getElementById(HTML_RUNNER_MOVEMENTS_TO_CHECK);
    runnerMovementsToCheckInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnerMovementsToCheckInput.onchange = runnerMovementsToCheckInputOnChange;
    cannonQueueInput = document.getElementById(HTML_CANNON_QUEUE);
    cannonQueueInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    healerCodesInput = document.getElementById(HTML_HEALER_CODES);
    healerCodesInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    healerSpawnTargetsInput = document.getElementById(HTML_HEALER_SPAWN_TARGETS);
    healerSpawnTargetsInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnerSpawnsInput = document.getElementById(HTML_RUNNER_SPAWNS);
    runnerSpawnsInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    healerSpawnsInput = document.getElementById(HTML_HEALER_SPAWNS);
    healerSpawnsInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnersDeadByTickInput = document.getElementById(HTML_RUNNERS_DEAD_BY_TICK);
    runnersDeadByTickInput.onkeydown = function (keyboardEvent) {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnersDeadByTickInput.onchange = runnersDeadByTickInputOnChange;
    simulateButton = document.getElementById(HTML_SIMULATE);
    simulateButton.onclick = simulateButtonOnClick;
    runnersDoNotDieWithMovements = document.getElementById(HTML_RUNNERS_DO_NOT_DIE_WITH_MOVEMENTS);
    const darkModeToggle = document.getElementById(HTML_TOGGLE_DARK_MODE);
    const savedDarkMode = localStorage.getItem("darkMode");
    if (savedDarkMode === "true") {
        document.body.classList.add("dark");
        darkModeToggle.checked = true;
    }
    darkModeToggle.onchange = function () {
        document.body.classList.toggle("dark", darkModeToggle.checked);
        localStorage.setItem("darkMode", String(darkModeToggle.checked));
    };
    document.getElementById(HTML_COPY_CONTROLLED_COMMANDS).onclick = function () {
        const text = document.getElementById(HTML_CONTROLLED_COMMANDS).innerText;
        navigator.clipboard.writeText(text);
    };
    document.getElementById(HTML_EXPORT_MARKERS).onclick = exportMarkers;
    document.getElementById(HTML_IMPORT_MARKERS).onclick = importMarkers;
    document.getElementById(HTML_SETTINGS_EXPORT).onclick = exportSettings;
    document.getElementById(HTML_SETTINGS_IMPORT).onclick = importSettings;
    pauseResumeButton.onclick = function () {
        if (!isRunning)
            return;
        isPaused = !isPaused;
        pauseResumeButton.innerHTML = isPaused ? "Resume" : "Pause";
    };
    stepBackButton.onclick = function () {
        if (!isRunning)
            return;
        isPaused = true;
        pauseResumeButton.innerHTML = "Resume";
        stepBackward();
    };
    stepForwardButton.onclick = function () {
        if (!isRunning)
            return;
        isPaused = true;
        pauseResumeButton.innerHTML = "Resume";
        stepForward();
    };
}
/**
 * Resets the simulator: the simulator is stopped and the underlying {@link BarbarianAssault} game
 * is replaced with a new game according to the currently selected configuration.
 */
function reset() {
    if (isRunning) {
        clearInterval(tickTimerId);
    }
    isRunning = false;
    startStopButton.innerHTML = "Start Wave";
    pauseResumeButton.style.display = "none";
    stepBackButton.style.display = "none";
    stepForwardButton.style.display = "none";
    document.getElementById(HTML_RUNNER_TABLE).style.display = "none";
    document.getElementById(HTML_HEALER_TABLE).style.display = "none";
    stateHistory = [];
    stateIndex = -1;
    barbarianAssault = new BarbarianAssault(wave, requireRepairs, requireLogs, infiniteFood, [], defenderLevel, player === "mainattacker" ? new Map : convertCommandsStringToMap(document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value, "mainattacker"), player === "secondattacker" ? new Map : convertCommandsStringToMap(document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value, "secondattacker"), player === "healer" ? new Map : convertCommandsStringToMap(document.getElementById(HTML_HEALER_COMMANDS).value, "healer"), player === "collector" ? new Map : convertCommandsStringToMap(document.getElementById(HTML_COLLECTOR_COMMANDS).value, "collector"), player === "defender" ? new Map : convertCommandsStringToMap(document.getElementById(HTML_DEFENDER_COMMANDS).value, "defender"), []);
    draw();
}
/**
 * Parses the simulator's configured runner movements, converting them into a list of per-runner
 * movement strings (each formatted e.g. as "wses" to indicate West-South-East-South).
 *
 * @return  a list of per-runner movements strings if the entire runner movements configuration
 *          is valid (i.e. contains only valid characters in the expected format), otherwise null
 */
function parseMovementsInput() {
    const movements = movementsInput.value.split("-");
    for (let i = 0; i < movements.length; i++) {
        const moves = movements[i];
        for (let j = 0; j < moves.length; j++) {
            const move = moves[j];
            if (move !== "" && move !== "s" && move !== "w" && move !== "e") {
                return null;
            }
        }
    }
    return movements;
}
function parseRunnerMovementsToCheck() {
    const runnerMovements = runnerMovementsToCheckInput.value.split("-");
    for (let i = 0; i < runnerMovements.length; i++) {
        const moves = runnerMovements[i];
        for (let j = 0; j < moves.length; j++) {
            const move = moves[j];
            if (move !== "" && move !== "s" && move !== "w" && move !== "e" && move !== "x") {
                return null;
            }
        }
    }
    return runnerMovements;
}
function parseFoodCallsInput() {
    const foodCalls = [];
    const foodCallsString = foodCallsInput.value;
    for (let i = 0; i < foodCallsString.length; i++) {
        switch (foodCallsString.charAt(i).toLowerCase()) {
            case "t":
                foodCalls.push(FoodType.TOFU);
                break;
            case "w":
                foodCalls.push(FoodType.WORMS);
                break;
            case "c":
                foodCalls.push(FoodType.CRACKERS);
                break;
            default:
                return null;
        }
    }
    return foodCalls;
}
function parseSpawnsInput(value) {
    const trimmed = value.trim();
    if (trimmed.length === 0) {
        return [];
    }
    const parts = trimmed.split(/[,\-]/);
    const spawns = [];
    for (const part of parts) {
        const tick = parseInt(part.trim(), 10);
        if (isNaN(tick) || tick < 1) {
            return [];
        }
        spawns.push(tick);
    }
    spawns.sort((a, b) => a - b);
    return spawns;
}
/**
 * Handles the given keyboard event.
 *
 * @param keyboardEvent the keyboard event to handle
 */
function windowOnKeyDown(keyboardEvent) {
    const key = keyboardEvent.key;
    if (isRunning) {
        if (simpleFood && player === "defender") {
            switch (key) {
                case "r":
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":t<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "w":
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":c<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "t":
                    barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":r<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "e":
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":1<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "l":
                    barbarianAssault.defenderPlayer.isPickingUpLogs = true;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":l<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
            }
        }
        switch (key) {
            case "t":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":t<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "c":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":c<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "w":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.WORMS);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":w<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "1":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":1<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "2":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.CRACKERS;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":2<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "3":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.WORMS;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":3<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "l":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.isPickingUpLogs = true;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":l<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "r":
                if (!simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":r<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "p":
                isPaused = !isPaused;
                pauseResumeButton.innerHTML = isPaused ? "Resume" : "Pause";
                break;
            case "s":
                if (isPaused || !pauseSaveLoad) {
                    isPaused = true;
                    pauseResumeButton.innerHTML = "Resume";
                    save();
                    saveExists = true;
                }
                break;
            case "y":
                if (saveExists && (isPaused || !pauseSaveLoad)) {
                    isPaused = true;
                    pauseResumeButton.innerHTML = "Resume";
                    load();
                }
                break;
            case "d":
                isPaused = true;
                pauseResumeButton.innerHTML = "Resume";
                stepBackward();
                break;
            case "f":
                isPaused = true;
                pauseResumeButton.innerHTML = "Resume";
                stepForward();
                break;
        }
    }
    if (key === " ") {
        startStopButtonOnClick();
        keyboardEvent.preventDefault();
    }
}
/**
 * Pauses and saves the state of the simulator.
 */
function save() {
    isPaused = true;
    savedBarbarianAssault = barbarianAssault.clone();
    savedTickCountSpanInnerHTML = tickCountSpan.innerHTML;
    savedCurrentDefenderFoodSpanInnerHTML = currentDefenderFoodSpan.innerHTML;
    savedPlayer = player;
    savedControlledCommandsInnerHTML = controlledCommands.innerHTML;
    savedDefenderLevel = defenderLevelSelection.value;
    savedWave = waveSelect.value;
    savedMovementsString = movementsInput.value;
    savedMainAttackerCommands = document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value;
    savedSecondAttackerCommands = document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value;
    savedHealerCommands = document.getElementById(HTML_HEALER_COMMANDS).value;
    savedCollectorCommands = document.getElementById(HTML_COLLECTOR_COMMANDS).value;
    savedDefenderCommands = document.getElementById(HTML_DEFENDER_COMMANDS).value;
    savedRequireRepairs = requireRepairs;
    savedInfiniteFood = infiniteFood;
    savedRequireLogs = requireLogs;
    savedFoodCallsString = foodCallsInput.value;
    savedCannonQueueString = cannonQueueInput.value;
    savedHealerCodesString = healerCodesInput.value;
    savedHealerSpawnTargetsString = healerSpawnTargetsInput.value;
    savedRunnerSpawnsString = runnerSpawnsInput.value;
    savedHealerSpawnsString = healerSpawnsInput.value;
}
/**
 * Pauses and loads the previously saved state of the simulator.
 */
function load() {
    isPaused = true;
    tickCountSpan.innerHTML = savedTickCountSpanInnerHTML;
    currentDefenderFoodSpan.innerHTML = savedCurrentDefenderFoodSpanInnerHTML;
    playerSelect.value = savedPlayer;
    player = savedPlayer;
    controlledCommands.innerHTML = savedControlledCommandsInnerHTML;
    defenderLevelSelection.value = savedDefenderLevel;
    defenderLevel = Number(defenderLevelSelection.value);
    waveSelect.value = savedWave;
    wave = Number(waveSelect.value);
    movementsInput.value = savedMovementsString;
    document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value = savedMainAttackerCommands;
    document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value = savedSecondAttackerCommands;
    document.getElementById(HTML_HEALER_COMMANDS).value = savedHealerCommands;
    document.getElementById(HTML_COLLECTOR_COMMANDS).value = savedCollectorCommands;
    document.getElementById(HTML_DEFENDER_COMMANDS).value = savedDefenderCommands;
    toggleLogToRepair.checked = savedRequireLogs;
    toggleRepair.checked = savedRequireRepairs;
    toggleInfiniteFood.checked = savedInfiniteFood;
    foodCallsInput.value = savedFoodCallsString;
    cannonQueueInput.value = savedCannonQueueString;
    healerCodesInput.value = savedHealerCodesString;
    healerSpawnTargetsInput.value = savedHealerSpawnTargetsString;
    runnerSpawnsInput.value = savedRunnerSpawnsString;
    healerSpawnsInput.value = savedHealerSpawnsString;
    barbarianAssault = savedBarbarianAssault;
    // the existing save state will mutate as the simulator proceeds,
    // so re-clone the save state in case of subsequent loads
    save();
    draw();
}
function pushState() {
    const snapshot = {
        ba: barbarianAssault.clone(),
        tickHTML: tickCountSpan.innerHTML,
        foodHTML: currentDefenderFoodSpan.innerHTML,
        commandsHTML: controlledCommands.innerHTML
    };
    stateHistory.splice(stateIndex + 1, Infinity, snapshot);
    stateIndex = stateHistory.length - 1;
    if (stateHistory.length > STATE_HISTORY_LIMIT) {
        const deleteCount = stateHistory.length - STATE_HISTORY_LIMIT;
        stateHistory.splice(0, deleteCount);
        stateIndex -= deleteCount;
    }
}
function stepBackward() {
    if (stateIndex <= 0) {
        return;
    }
    stateIndex--;
    loadState(stateHistory[stateIndex]);
}
function stepForward() {
    if (stateIndex >= stateHistory.length - 1) {
        return;
    }
    stateIndex++;
    loadState(stateHistory[stateIndex]);
}
function loadState(snapshot) {
    barbarianAssault = snapshot.ba.clone();
    tickCountSpan.innerHTML = snapshot.tickHTML;
    currentDefenderFoodSpan.innerHTML = snapshot.foodHTML;
    controlledCommands.innerHTML = snapshot.commandsHTML;
    draw();
    updateRunnerTable();
    updateHealerTable();
}
/**
 * Handles the given mouse event.
 *
 * @param mouseEvent    the mouse event to handle
 */
function canvasOnMouseDown(mouseEvent) {
    const canvasRect = renderer.canvas.getBoundingClientRect();
    const xTile = Math.trunc((mouseEvent.clientX - canvasRect.left) / renderer.tileSize);
    const yTile = Math.trunc((canvasRect.bottom - 1 - mouseEvent.clientY) / renderer.tileSize);
    if (mouseEvent.button === 0) {
        if (markingTiles) {
            let tileAlreadyMarked = false;
            for (let i = 0; i < markedTiles.length; i++) {
                if ((markedTiles[i].position.x === xTile) && (markedTiles[i].position.y === yTile)) {
                    tileAlreadyMarked = true;
                    markedTiles.splice(i, 1);
                }
            }
            if (!tileAlreadyMarked) {
                markedTiles.push(new TileMarker(new Position(xTile, yTile), RGBColor.fromHexColor(markerColor)));
            }
            if (!isRunning) {
                draw();
            }
        }
        else {
            switch (player) {
                case "defender":
                    barbarianAssault.defenderPlayer.findPath(barbarianAssault, new Position(xTile, yTile));
                    break;
                case "mainattacker":
                    barbarianAssault.mainAttackerPlayer.findPath(barbarianAssault, new Position(xTile, yTile));
                    break;
                case "secondattacker":
                    barbarianAssault.secondAttackerPlayer.findPath(barbarianAssault, new Position(xTile, yTile));
                    break;
                case "healer":
                    barbarianAssault.healerPlayer.findPath(barbarianAssault, new Position(xTile, yTile));
                    break;
                case "collector":
                    barbarianAssault.collectorPlayer.findPath(barbarianAssault, new Position(xTile, yTile));
                    break;
                default:
                    break;
            }
            controlledCommands.innerHTML += barbarianAssault.ticks + ":" + xTile + "," + yTile + "<br>";
            controlledCommands.scrollTop = controlledCommands.scrollHeight;
        }
    }
}
/**
 * Draws and presents the entire display of the simulator.
 */
function draw() {
    drawMap();
    drawDetails();
    drawItems();
    drawEntities();
    drawGrid();
    drawOverlays();
    renderer.present();
}
/**
 * Draws the map.
 */
function drawMap() {
    renderer.setDrawColor(206, 183, 117, 255);
    renderer.clear();
    for (let y = 0; y < barbarianAssault.map.height; y++) {
        for (let x = 0; x < barbarianAssault.map.width; x++) {
            const flag = barbarianAssault.map.getFlag(new Position(x, y));
            if ((flag & LOS_FULL_MASK) !== 0) {
                renderer.setDrawColor(0, 0, 0, 255);
                renderer.fillOpaque(x, y);
            }
            else {
                if ((flag & MOVE_FULL_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.fillOpaque(x, y);
                }
                if ((flag & LOS_EAST_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.eastLine(x, y);
                }
                else if ((flag & MOVE_EAST_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.eastLine(x, y);
                }
                if ((flag & LOS_WEST_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.westLine(x, y);
                }
                else if ((flag & MOVE_WEST_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.westLine(x, y);
                }
                if ((flag & LOS_NORTH_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.northLine(x, y);
                }
                else if ((flag & MOVE_NORTH_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.northLine(x, y);
                }
                if ((flag & LOS_SOUTH_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.southLine(x, y);
                }
                else if ((flag & MOVE_SOUTH_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.southLine(x, y);
                }
            }
        }
    }
    renderer.setDrawColor((markerColor >> 16) & 255, (markerColor >> 8) & 255, markerColor & 255, 255);
    for (let i = 0; i < markedTiles.length; i++) {
        const markedTile = markedTiles[i];
        renderer.setDrawColor(markedTile.rgbColor.r, markedTile.rgbColor.g, markedTile.rgbColor.b, 255);
        renderer.outline(markedTile.position.x, markedTile.position.y);
    }
}
/**
 * Draws details of the game (aesthetic details, logs, and traps).
 */
function drawDetails() {
    renderer.setDrawColor(160, 82, 45, 255);
    renderer.cone(40, 32);
    renderer.cone(40, 31);
    renderer.cone(41, 32);
    renderer.cone(41, 31);
    renderer.cone(43, 31);
    renderer.cone(36, 34);
    renderer.cone(36, 35);
    renderer.cone(37, 34);
    renderer.cone(37, 35);
    renderer.cone(39, 36);
    renderer.cone(43, 22);
    renderer.cone(43, 23);
    renderer.cone(44, 22);
    renderer.cone(44, 23);
    renderer.cone(45, 24);
    if (barbarianAssault.southeastLogsArePresent) {
        renderer.fillItem(barbarianAssault.southeastLogsPosition.x, barbarianAssault.southeastLogsPosition.y);
    }
    if (barbarianAssault.northwestLogsArePresent) {
        renderer.fillItem(barbarianAssault.northwestLogsPosition.x, barbarianAssault.northwestLogsPosition.y);
    }
    if (barbarianAssault.eastTrapCharges < 1) {
        renderer.setDrawColor(255, 0, 0, 255);
    }
    else if (barbarianAssault.eastTrapCharges === 1) {
        renderer.setDrawColor(255, 140, 0, 255);
    }
    renderer.fill(45, 26);
    renderer.setDrawColor(160, 82, 45, 255);
    if (barbarianAssault.westTrapCharges < 1) {
        renderer.setDrawColor(255, 0, 0, 255);
    }
    else if (barbarianAssault.westTrapCharges === 1) {
        renderer.setDrawColor(255, 140, 0, 255);
    }
    renderer.fill(15, 25);
    renderer.setDrawColor(160, 82, 45, 255);
    // queen trapdoor
    if (wave === 10) {
        renderer.outlineBig(27, 20, 8, 8);
    }
    renderer.setDrawColor(127, 127, 127, 255);
    renderer.fillItem(32, 34);
    // Draw cannons
    const westCannon = getCannonPosition(CannonSide.WEST);
    const eastCannon = getCannonPosition(CannonSide.EAST);
    renderer.setDrawColor(80, 80, 80, 200);
    renderer.fill(westCannon.x, westCannon.y);
    renderer.fill(eastCannon.x, eastCannon.y);
    renderer.setDrawColor(40, 40, 40, 255);
    renderer.outline(westCannon.x, westCannon.y);
    renderer.outline(eastCannon.x, eastCannon.y);
}
/**
 * Draws items (e.g. {@link Food}.
 */
function drawItems() {
    for (let i = 0; i < barbarianAssault.map.foodZones.length; i++) {
        const foodZone = barbarianAssault.map.foodZones[i];
        for (let j = 0; j < foodZone.foodList.length; j++) {
            const food = foodZone.foodList[j];
            renderer.setDrawColor(food.colorRed, food.colorGreen, food.colorBlue, 127);
            renderer.fillItem(food.position.x, food.position.y);
        }
    }
}
/**
 * Draws entities ({@link Character}s}.
 */
function drawEntities() {
    barbarianAssault.runners.forEach((runner) => {
        if (runner.blueCounter >= 0) {
            renderer.setDrawColor(10, 10, 240, 200);
            renderer.fill(runner.position.x, runner.position.y);
            renderer.setDrawColor(100, 100, 255, 255);
            renderer.outline(runner.position.x, runner.position.y);
        }
        else if (runner.greenCounter >= 0) {
            renderer.setDrawColor(10, 10, 240, 127);
            renderer.fill(runner.position.x, runner.position.y);
            renderer.setDrawColor(0, 200, 0, 255);
            renderer.outline(runner.position.x, runner.position.y);
        }
        else if (runner.isDying) {
            renderer.setDrawColor(5, 5, 120, 127);
            renderer.fill(runner.position.x, runner.position.y);
        }
        else {
            renderer.setDrawColor(10, 10, 240, 127);
            renderer.fill(runner.position.x, runner.position.y);
        }
    });
    barbarianAssault.healers.forEach((healer) => {
        if (healer.zombieState) {
            renderer.setDrawColor(80, 80, 80, 180);
            renderer.fill(healer.position.x, healer.position.y);
            renderer.setDrawColor(0, 200, 0, 255);
            renderer.outline(healer.position.x, healer.position.y);
        }
        else if (healer.blueCounter >= 0) {
            renderer.setDrawColor(10, 240, 10, 127);
            renderer.fill(healer.position.x, healer.position.y);
            renderer.setDrawColor(100, 100, 255, 255);
            renderer.outline(healer.position.x, healer.position.y);
        }
        else if (healer.isDying) {
            renderer.setDrawColor(5, 120, 5, 127);
            renderer.fill(healer.position.x, healer.position.y);
        }
        else {
            renderer.setDrawColor(10, 240, 10, 127);
            renderer.fill(healer.position.x, healer.position.y);
        }
    });
    if (barbarianAssault.collectorPlayer.position.x >= 0 && barbarianAssault.collectorPlayer.position.y >= 0) {
        renderer.setDrawColor(240, 240, 10, 220);
        renderer.fill(barbarianAssault.collectorPlayer.position.x, barbarianAssault.collectorPlayer.position.y);
        renderer.setDrawColor(0, 0, 0, 220);
        renderer.outline(barbarianAssault.collectorPlayer.position.x, barbarianAssault.collectorPlayer.position.y);
    }
    if (barbarianAssault.mainAttackerPlayer.position.x >= 0 && barbarianAssault.mainAttackerPlayer.position.y >= 0) {
        renderer.setDrawColor(240, 10, 10, 220);
        renderer.fill(barbarianAssault.mainAttackerPlayer.position.x, barbarianAssault.mainAttackerPlayer.position.y);
        renderer.setDrawColor(0, 0, 0, 220);
        renderer.outline(barbarianAssault.mainAttackerPlayer.position.x, barbarianAssault.mainAttackerPlayer.position.y);
    }
    if (barbarianAssault.secondAttackerPlayer.position.x >= 0 && barbarianAssault.secondAttackerPlayer.position.y >= 0) {
        renderer.setDrawColor(200, 50, 50, 220);
        renderer.fill(barbarianAssault.secondAttackerPlayer.position.x, barbarianAssault.secondAttackerPlayer.position.y);
        renderer.setDrawColor(0, 0, 0, 220);
        renderer.outline(barbarianAssault.secondAttackerPlayer.position.x, barbarianAssault.secondAttackerPlayer.position.y);
    }
    if (barbarianAssault.healerPlayer.position.x >= 0 && barbarianAssault.healerPlayer.position.y >= 0) {
        renderer.setDrawColor(10, 240, 10, 220);
        renderer.fill(barbarianAssault.healerPlayer.position.x, barbarianAssault.healerPlayer.position.y);
        renderer.setDrawColor(0, 0, 0, 220);
        renderer.outline(barbarianAssault.healerPlayer.position.x, barbarianAssault.healerPlayer.position.y);
    }
    if (barbarianAssault.defenderPlayer.position.x >= 0 && barbarianAssault.defenderPlayer.position.y >= 0) {
        renderer.setDrawColor(240, 240, 240, 220);
        renderer.fill(barbarianAssault.defenderPlayer.position.x, barbarianAssault.defenderPlayer.position.y);
    }
}
/**
 * Draws a grid, with each tile of the map being a cell of the grid (i.e. outlines each tile).
 */
function drawGrid() {
    for (let xTile = 0; xTile < barbarianAssault.map.width; xTile++) {
        if (xTile % 8 === 7) {
            renderer.setDrawColor(40, 22, 12, 175);
        }
        else {
            renderer.setDrawColor(0, 0, 0, 48);
        }
        renderer.eastLineBig(xTile, 0, barbarianAssault.map.height);
    }
    for (let yTile = 0; yTile < barbarianAssault.map.height; yTile++) {
        if (yTile % 8 === 7) {
            renderer.setDrawColor(40, 22, 12, 175);
        }
        else {
            renderer.setDrawColor(0, 0, 0, 48);
        }
        renderer.northLineBig(0, yTile, barbarianAssault.map.width);
    }
}
/**
 * Draws aesthetic overlays.
 */
function drawOverlays() {
    renderer.setDrawColor(240, 10, 10, 220);
    if (wave === 10) {
        renderer.outline(18, 38);
    }
    else {
        renderer.outline(18, 37);
    }
    renderer.outline(24, 39);
    renderer.fill(33, 6);
    renderer.setDrawColor(10, 10, 240, 220);
    if (wave === 10) {
        renderer.outline(42, 38);
    }
    else {
        renderer.outline(36, 39);
    }
    renderer.fill(34, 6);
    renderer.setDrawColor(10, 240, 10, 220);
    if (wave === 10) {
        renderer.outline(36, 39);
    }
    else {
        renderer.outline(42, 37);
    }
    renderer.fill(35, 6);
    renderer.setDrawColor(240, 240, 10, 220);
    ;
    renderer.fill(36, 6);
    if (toggleRenderDistance.checked) {
        drawRenderDistance();
    }
}
function drawRenderDistance() {
    const players = [
        barbarianAssault.defenderPlayer,
        barbarianAssault.collectorPlayer,
        barbarianAssault.mainAttackerPlayer,
        barbarianAssault.secondAttackerPlayer,
        barbarianAssault.healerPlayer
    ];
    renderer.setDrawColor(0, 0, 0, 80);
    for (let x = 0; x < barbarianAssault.map.width; x++) {
        for (let y = 0; y < barbarianAssault.map.height; y++) {
            let inRange = false;
            for (const p of players) {
                if (Math.max(Math.abs(x - p.position.x), Math.abs(y - p.position.y)) <= 15) {
                    inRange = true;
                    break;
                }
            }
            if (!inRange) {
                renderer.fill(x, y);
            }
        }
    }
}
/**
 * If the simulator is running, then stops and resets the simulator.
 * Otherwise, starts the simulator.
 */
function startStopButtonOnClick() {
    if (isRunning) {
        barbarianAssault.map.reset();
        reset();
    }
    else {
        const movements = parseMovementsInput();
        if (movements === null) {
            alert("Invalid runner movements. Example: ws-s");
            return;
        }
        const foodCalls = simpleFood ? [] : parseFoodCallsInput();
        if (foodCalls === null) {
            alert("Invalid food calls. Example: twcw");
            return;
        }
        const mainAttackerCommands = convertCommandsStringToMap(document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value, "mainattacker");
        const secondAttackerCommands = convertCommandsStringToMap(document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value, "secondattacker");
        const healerCommands = convertCommandsStringToMap(document.getElementById(HTML_HEALER_COMMANDS).value, "healer");
        const collectorCommands = convertCommandsStringToMap(document.getElementById(HTML_COLLECTOR_COMMANDS).value, "collector");
        const defenderCommands = convertCommandsStringToMap(document.getElementById(HTML_DEFENDER_COMMANDS).value, "defender");
        if (mainAttackerCommands === null || secondAttackerCommands === null || healerCommands === null || collectorCommands === null || defenderCommands === null) {
            alert("Invalid team commands. Example: 7:20,24");
            return;
        }
        const cannonQueue = parseCannonInput(cannonQueueInput.value);
        if (cannonQueue === null) {
            alert("Invalid cannon queue. Example: wrr,1,51-erg,1,21");
            return;
        }
        let healerCodeActions = [];
        try {
            healerCodeActions = parseHealerCodes(healerCodesInput.value);
            assignSpawnPriorities(healerCodeActions);
        }
        catch (e) {
            alert("Invalid healer codes. Example: h1,3-h2,5-h3,4");
            return;
        }
        isRunning = true;
        isPaused = false;
        startStopButton.innerHTML = "Stop Wave";
        pauseResumeButton.innerHTML = "Pause";
        pauseResumeButton.style.display = "";
        stepBackButton.style.display = "";
        stepForwardButton.style.display = "";
        controlledCommands.innerHTML = "";
        barbarianAssault = new BarbarianAssault(wave, requireRepairs, requireLogs, simpleFood ? true : infiniteFood, movements, defenderLevel, player === "mainattacker" ? new Map : mainAttackerCommands, player === "secondattacker" ? new Map : secondAttackerCommands, player === "healer" ? new Map : healerCommands, player === "collector" ? new Map : collectorCommands, player === "defender" ? new Map : defenderCommands, foodCalls, cannonQueue);
        if (healerCodeActions.length > 0) {
            barbarianAssault.healerPlayer.codeQueue = healerCodeActions;
        }
        const spawnTargetsValue = healerSpawnTargetsInput.value.trim();
        if (spawnTargetsValue.length > 0) {
            barbarianAssault.healerSpawnTargets = spawnTargetsValue.split("-");
        }
        barbarianAssault.simpleFood = simpleFood;
        barbarianAssault.runnerSpawns = parseSpawnsInput(runnerSpawnsInput.value);
        barbarianAssault.healerSpawns = parseSpawnsInput(healerSpawnsInput.value);
        barbarianAssault.renderDistanceEnabled = toggleRenderDistance.checked;
        console.log("Wave " + wave + " started!");
        tick();
        tickTimerId = setInterval(tick, Number(tickDurationInput.value));
    }
}
function simulateButtonOnClick() {
    if (isRunning) {
        barbarianAssault.map.reset();
        reset();
    }
    const runnerMovementsToCheck = parseRunnerMovementsToCheck();
    if (runnerMovementsToCheck === null) {
        alert("Invalid runner movements to check. Example: ws-x-ex");
        return;
    }
    const foodCalls = parseFoodCallsInput();
    if (foodCalls === null) {
        alert("Invalid food calls. Example: twcw");
        return;
    }
    const runnersDeadByTick = Number(runnersDeadByTickInput.value);
    if (!Number.isInteger(runnersDeadByTick) || runnersDeadByTick < 1) {
        alert("Invalid runners dead by tick. Example: 12");
        return;
    }
    runnersDoNotDieWithMovements.innerHTML = "";
    startStopButton.disabled = true;
    movementsInput.disabled = true;
    foodCallsInput.disabled = true;
    toggleInfiniteFood.disabled = true;
    toggleRepair.disabled = true;
    toggleLogToRepair.disabled = true;
    runnerMovementsToCheckInput.disabled = true;
    runnersDeadByTickInput.disabled = true;
    simulateButton.disabled = true;
    const movementsRunnersDoNotDieOnTime = getMovementsRunnersDoNotDieOnTime(foodCalls, runnerMovementsToCheck, runnersDeadByTick);
    let runnersDoNotDieWithMovementsInnerHTML = "";
    for (let i = 0; i < movementsRunnersDoNotDieOnTime.length; i++) {
        const movement = movementsRunnersDoNotDieOnTime[i];
        runnersDoNotDieWithMovementsInnerHTML += getMovementsStringFromArray(movement) + "<br>";
    }
    runnersDoNotDieWithMovements.innerHTML = runnersDoNotDieWithMovementsInnerHTML;
    startStopButton.disabled = false;
    movementsInput.disabled = false;
    foodCallsInput.disabled = false;
    toggleInfiniteFood.disabled = false;
    toggleRepair.disabled = false;
    toggleLogToRepair.disabled = false;
    runnerMovementsToCheckInput.disabled = false;
    runnersDeadByTickInput.disabled = false;
    simulateButton.disabled = false;
}
/**
 * Progresses the state of the simulator by a single tick.
 */
function updateRunnerTable() {
    const table = document.getElementById(HTML_RUNNER_TABLE);
    if (!isRunning) {
        table.style.display = "none";
        return;
    }
    table.style.display = "table";
    const oldBody = table.querySelector("tbody");
    if (oldBody) {
        oldBody.remove();
    }
    const tbody = document.createElement("tbody");
    barbarianAssault.runners.forEach((runner) => {
        const row = tbody.insertRow();
        const cellStyle = "border: 1px solid; padding: 2px 6px;";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.id}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.cycleTick}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.targetState}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.hp}/5</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${runner.position.x}, ${runner.position.y})</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${runner.destination.x}, ${runner.destination.y})</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.foodTarget !== null ? "(" + runner.foodTarget.position.x + ", " + runner.foodTarget.position.y + ")" : "None"}</td>`;
        let status = "";
        if (runner.isDying)
            status = "Dying";
        else if (runner.blueCounter >= 0)
            status = "Stunned";
        else if (runner.greenCounter >= 0)
            status = "Poisoned";
        else if (runner.blughhhhCountdown > 0)
            status = "Blughhhh";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${status}</td>`;
    });
    table.appendChild(tbody);
}
function updateHealerTable() {
    const table = document.getElementById(HTML_HEALER_TABLE);
    if (!isRunning) {
        table.style.display = "none";
        return;
    }
    table.style.display = "table";
    const oldBody = table.querySelector("tbody");
    if (oldBody) {
        oldBody.remove();
    }
    const tbody = document.createElement("tbody");
    barbarianAssault.healers.forEach((healer) => {
        const row = tbody.insertRow();
        const cellStyle = "border: 1px solid; padding: 2px 6px;";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.id}</td>`;
        let targetStr = "";
        if (healer.target instanceof Player) {
            if (healer.target === barbarianAssault.mainAttackerPlayer)
                targetStr = "Main";
            else if (healer.target === barbarianAssault.secondAttackerPlayer)
                targetStr = "Second";
            else if (healer.target === barbarianAssault.healerPlayer)
                targetStr = "Healer";
            else if (healer.target === barbarianAssault.collectorPlayer)
                targetStr = "Collector";
            else if (healer.target === barbarianAssault.defenderPlayer)
                targetStr = "Defender";
        }
        else if (healer.target instanceof RunnerPenance) {
            targetStr = "Runner";
        }
        row.insertCell().outerHTML = `<td style="${cellStyle}">${targetStr}</td>`;
        let lastTargetStr = "";
        if (healer.previousTargetType === HealerTargetType.PLAYER)
            lastTargetStr = "Player";
        else if (healer.previousTargetType === HealerTargetType.RUNNER)
            lastTargetStr = "Runner";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${lastTargetStr}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.sprayTimer}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.isPoisoned ? "Yes" : "No"}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.health}/${healer.maxHealth}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${healer.position.x}, ${healer.position.y})</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${healer.destination.x}, ${healer.destination.y})</td>`;
        let status = "";
        if (healer.isDying)
            status = "Dying";
        else if (healer.zombieState)
            status = "Zombie";
        else if (healer.blueCounter >= 0)
            status = "Stunned";
        else if (healer.greenCounter >= 0)
            status = "Egg Psn";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${status}</td>`;
    });
    table.appendChild(tbody);
}
function tick() {
    if (!isPaused) {
        barbarianAssault.tick();
        if (!simpleFood) {
            currentDefenderFoodSpan.innerHTML = barbarianAssault.defenderFoodCall.toString();
        }
        tickCountSpan.innerHTML = barbarianAssault.ticks.toString() + " (" + ticksToSeconds(barbarianAssault.ticks) + "s)";
        pushState();
        draw();
        updateRunnerTable();
        updateHealerTable();
    }
}
/**
 * Toggles whether tile-marking mode is enabled.
 */
function toggleMarkingTilesOnChange() {
    markingTiles = toggleMarkingTiles.checked;
}
function exportMarkers() {
    const regionId = wave === 10 ? 7508 : 7509;
    const tiles = markedTiles.map((tile) => ({
        regionId: regionId,
        regionX: tile.position.x,
        regionY: tile.position.y + 8,
        z: 0,
        color: "#ff" + rgbToHex(tile.rgbColor)
    }));
    const json = JSON.stringify(tiles);
    navigator.clipboard.writeText(json);
    alert("Copied " + tiles.length + " tile marker(s) to clipboard.");
}
function importMarkers() {
    const field = document.getElementById(HTML_MARKER_IMPORT_FIELD);
    const input = field.value.trim();
    if (input.length === 0) {
        return;
    }
    try {
        const tiles = JSON.parse(input);
        const regionId = wave === 10 ? 7508 : 7509;
        let count = 0;
        for (const tile of tiles) {
            if (tile.regionId !== regionId) {
                continue;
            }
            const x = tile.regionX;
            const y = tile.regionY - 8;
            const colorHex = "#" + tile.color.slice(3);
            const hexNum = Number("0x" + colorHex.substring(1));
            const color = RGBColor.fromHexColor(hexNum);
            // remove existing marker at same position
            for (let i = markedTiles.length - 1; i >= 0; i--) {
                if (markedTiles[i].position.x === x && markedTiles[i].position.y === y) {
                    markedTiles.splice(i, 1);
                }
            }
            markedTiles.push(new TileMarker(new Position(x, y), color));
            count++;
        }
        field.value = "";
        alert("Imported " + count + " tile marker(s).");
        if (!isRunning) {
            draw();
        }
    }
    catch (e) {
        alert("Failed to parse RuneLite tile marker JSON.");
    }
}
function exportSettings() {
    const settings = {
        defenderLevel: defenderLevelSelection.value,
        wave: waveSelect.value,
        runnerMovements: movementsInput.value,
        foodCalls: foodCallsInput.value,
        cannonQueue: cannonQueueInput.value,
        healerCodes: healerCodesInput.value,
        healerSpawnTargets: healerSpawnTargetsInput.value,
        runnerSpawns: runnerSpawnsInput.value,
        healerSpawns: healerSpawnsInput.value,
        tickDuration: document.getElementById(HTML_TICK_DURATION).value,
        infiniteFood: document.getElementById(HTML_TOGGLE_INFINITE_FOOD).checked,
        requireRepairs: document.getElementById(HTML_TOGGLE_REPAIR).checked,
        requireLogToRepair: document.getElementById(HTML_TOGGLE_LOG_TO_REPAIR).checked,
        renderDistance: document.getElementById(HTML_TOGGLE_RENDER_DISTANCE).checked,
        simpleFood: simpleFood,
        mainAttacker: document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value,
        secondAttacker: document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value,
        healer: document.getElementById(HTML_HEALER_COMMANDS).value,
        collector: document.getElementById(HTML_COLLECTOR_COMMANDS).value,
        defender: document.getElementById(HTML_DEFENDER_COMMANDS).value,
        playerToControl: playerSelect.value,
    };
    const json = JSON.stringify(settings);
    navigator.clipboard.writeText(json);
    alert("Settings copied to clipboard.");
}
function importSettings() {
    const field = document.getElementById(HTML_SETTINGS_IMPORT_FIELD);
    const input = field.value.trim();
    if (input.length === 0) {
        return;
    }
    try {
        const s = JSON.parse(input);
        if (s.defenderLevel !== undefined)
            defenderLevelSelection.value = s.defenderLevel;
        if (s.wave !== undefined) {
            waveSelect.value = s.wave;
            wave = Number(waveSelect.value);
        }
        if (s.runnerMovements !== undefined)
            movementsInput.value = s.runnerMovements;
        if (s.foodCalls !== undefined)
            foodCallsInput.value = s.foodCalls;
        if (s.cannonQueue !== undefined)
            cannonQueueInput.value = s.cannonQueue;
        if (s.healerCodes !== undefined)
            healerCodesInput.value = s.healerCodes;
        if (s.healerSpawnTargets !== undefined)
            healerSpawnTargetsInput.value = s.healerSpawnTargets;
        if (s.runnerSpawns !== undefined)
            runnerSpawnsInput.value = s.runnerSpawns;
        if (s.healerSpawns !== undefined)
            healerSpawnsInput.value = s.healerSpawns;
        if (s.tickDuration !== undefined)
            document.getElementById(HTML_TICK_DURATION).value = s.tickDuration;
        if (s.infiniteFood !== undefined)
            document.getElementById(HTML_TOGGLE_INFINITE_FOOD).checked = s.infiniteFood;
        if (s.requireRepairs !== undefined)
            document.getElementById(HTML_TOGGLE_REPAIR).checked = s.requireRepairs;
        if (s.requireLogToRepair !== undefined)
            document.getElementById(HTML_TOGGLE_LOG_TO_REPAIR).checked = s.requireLogToRepair;
        if (s.renderDistance !== undefined)
            document.getElementById(HTML_TOGGLE_RENDER_DISTANCE).checked = s.renderDistance;
        if (s.simpleFood !== undefined && s.simpleFood !== simpleFood) {
            document.getElementById(HTML_TOGGLE_SIMPLE_FOOD).click();
        }
        if (s.mainAttacker !== undefined)
            document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value = s.mainAttacker;
        if (s.secondAttacker !== undefined)
            document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value = s.secondAttacker;
        if (s.healer !== undefined)
            document.getElementById(HTML_HEALER_COMMANDS).value = s.healer;
        if (s.collector !== undefined)
            document.getElementById(HTML_COLLECTOR_COMMANDS).value = s.collector;
        if (s.defender !== undefined)
            document.getElementById(HTML_DEFENDER_COMMANDS).value = s.defender;
        if (s.playerToControl !== undefined) {
            playerSelect.value = s.playerToControl;
            player = playerSelect.value;
        }
        field.value = "";
        alert("Settings imported.");
    }
    catch (e) {
        alert("Failed to parse settings JSON.");
    }
}
function rgbToHex(color) {
    return ((1 << 24) | (color.r << 16) | (color.g << 8) | color.b).toString(16).slice(1);
}
/**
 * Sets the wave to the selected wave value, and stops and resets the simulator.
 */
function waveSelectOnChange() {
    wave = Number(waveSelect.value);
    reset();
}
function playerSelectOnChange() {
    player = playerSelect.value;
    reset();
}
/**
 * Sets the defender level to the selected defender level value, and stops and resets the simulator.
 */
function defenderLevelSelectionOnChange() {
    defenderLevel = Number(defenderLevelSelection.value);
    reset();
}
/**
 * Toggles whether traps need to be repaired.
 */
function toggleRepairOnChange() {
    requireRepairs = toggleRepair.checked;
    reset();
}
function movementsInputOnChange() {
    reset();
}
function runnerMovementsToCheckInputOnChange() {
    reset();
}
function runnersDeadByTickInputOnChange() {
    reset();
}
function foodCallsInputOnChange() {
    reset();
}
function markerColorInputOnChange() {
    markerColor = Number("0x" + markerColorInput.value.substring(1));
}
/**
 * Toggles whether the simulator must be paused before saving / loading.
 */
function togglePauseSaveLoadOnChange() {
    pauseSaveLoad = togglePauseSaveLoad.checked;
}
/**
 * Toggles whether the defender has infinite food.
 */
function toggleInfiniteFoodOnChange() {
    infiniteFood = toggleInfiniteFood.checked;
    reset();
}
/**
 * Toggles whether a log is required to repair a trap.
 */
function toggleLogToRepairOnChange() {
    requireLogs = toggleLogToRepair.checked;
    reset();
}
/**
 * Converts the given commands string to a map from tick numbers to positions.
 * If the given commands string is invalid, then null is returned
 *
 * @param commandsString    the commands string to convert to a map from tick
 *                          numbers to positions
 * @return                  a map from tick numbers to positions as specified by
 *                          the given commands string, or null if the given
 *                          commands string is invalid
 */
function convertCommandsStringToMap(commandsString, player) {
    if (commandsString === null) {
        return null;
    }
    const commandsMap = new Map();
    const commands = commandsString.split("\n");
    let previousCommandTick = -1;
    for (let i = 0; i < commands.length; i++) {
        const command = commands[i];
        if (command.length === 0) {
            continue;
        }
        const tickAndCommand = command.split(":");
        if (tickAndCommand.length !== 2) {
            return null;
        }
        const tick = Number(tickAndCommand[0]);
        if (!Number.isInteger(tick) || tick < 1 || tick < previousCommandTick) {
            return null;
        }
        const commandTokens = tickAndCommand[1].split(",");
        if (commandTokens.length === 1) {
            if (player !== "defender") {
                return null;
            }
            switch (commandTokens[0]) {
                case "t":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.DROP_TOFU));
                    break;
                case "c":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.DROP_CRACKERS));
                    break;
                case "w":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.DROP_WORMS));
                    break;
                case "1":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_TOFU));
                    break;
                case "2":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_CRACKERS));
                    break;
                case "3":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_WORMS));
                    break;
                case "l":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_LOGS));
                    break;
                case "r":
                    addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.REPAIR_TRAP));
                    break;
                default:
                    return null;
            }
        }
        else if (commandTokens.length === 2) {
            const positionX = Number(commandTokens[0]);
            const positionY = Number(commandTokens[1]);
            if (!Number.isInteger(positionX) || !Number.isInteger(positionY)) {
                return null;
            }
            addToCommandsMap(commandsMap, tick, new MoveCommand(new Position(positionX, positionY)));
        }
        else {
            return null;
        }
        previousCommandTick = tick;
    }
    return commandsMap;
}
function addToCommandsMap(commandsMap, tick, command) {
    if (commandsMap.has(tick)) {
        commandsMap.get(tick).push(command);
    }
    else {
        commandsMap.set(tick, [command]);
    }
}
/**
 * Converts time measured in ticks to seconds.
 *
 * @param ticks the number of ticks to convert to seconds
 */
function ticksToSeconds(ticks) {
    return (0.6 * Math.max(ticks - 1, 0)).toFixed(1);
}
function runnersDieOnTimeForMovements(runnerMovements, foodCalls, runnersDeadByTick, mainAttackerCommands, secondAttackerCommands, healerCommands, collectorCommands, defenderCommands) {
    const cannonQueue = parseCannonInput(cannonQueueInput.value);
    const barbarianAssaultSim = new BarbarianAssault(wave, requireRepairs, requireLogs, infiniteFood, runnerMovements, defenderLevel, mainAttackerCommands, secondAttackerCommands, healerCommands, collectorCommands, defenderCommands, foodCalls, cannonQueue || []);
    barbarianAssaultSim.simpleFood = simpleFood;
    barbarianAssaultSim.runnerSpawns = parseSpawnsInput(runnerSpawnsInput.value);
    barbarianAssaultSim.healerSpawns = parseSpawnsInput(healerSpawnsInput.value);
    for (let i = 0; i < runnersDeadByTick; i++) {
        barbarianAssaultSim.tick();
    }
    return barbarianAssaultSim.runnersKilled === barbarianAssaultSim.totalRunners;
}
function getMovementsRunnersDoNotDieOnTime(foodCalls, runnerMovementsToCheck, runnersDeadByTick) {
    const movementsRunnersDoNotDieOnTime = [];
    const candidateMovements = [];
    runnerMovementsToCheck.forEach((movementPattern) => {
        candidateMovements.push(getAllForcedMovementsForOneRunner(movementPattern));
    });
    const allCombinations = getAllCombinations(candidateMovements, 0, [[]]);
    const mainAttackerCommands = convertCommandsStringToMap(document.getElementById(HTML_MAIN_ATTACKER_COMMANDS).value, "mainattacker");
    const secondAttackerCommands = convertCommandsStringToMap(document.getElementById(HTML_SECOND_ATTACKER_COMMANDS).value, "secondattacker");
    const healerCommands = convertCommandsStringToMap(document.getElementById(HTML_HEALER_COMMANDS).value, "healer");
    const collectorCommands = convertCommandsStringToMap(document.getElementById(HTML_COLLECTOR_COMMANDS).value, "collector");
    const defenderCommands = convertCommandsStringToMap(document.getElementById(HTML_DEFENDER_COMMANDS).value, "defender");
    for (let i = 0; i < allCombinations.length; i++) {
        const runnerMovements = allCombinations[i];
        if (!runnersDieOnTimeForMovements(runnerMovements, foodCalls, runnersDeadByTick, mainAttackerCommands, secondAttackerCommands, healerCommands, collectorCommands, defenderCommands)) {
            movementsRunnersDoNotDieOnTime.push(runnerMovements);
        }
    }
    return movementsRunnersDoNotDieOnTime;
}
function getAllCombinations(candidateMovements, index, partialCombinations) {
    if (index >= candidateMovements.length) {
        return partialCombinations;
    }
    const newPartialCombinations = [];
    partialCombinations.forEach((partialCombination) => {
        candidateMovements[index].forEach((candidateMovement) => {
            newPartialCombinations.push([...partialCombination, candidateMovement]);
        });
    });
    return getAllCombinations(candidateMovements, index + 1, newPartialCombinations);
}
function getAllForcedMovementsForOneRunner(movementPattern) {
    const validDirections = [];
    for (let i = 0; i < movementPattern.length; i++) {
        switch (movementPattern.charAt(i)) {
            case "w":
                validDirections.push(["w"]);
                break;
            case "e":
                validDirections.push(["e"]);
                break;
            case "s":
                validDirections.push(["s"]);
                break;
            case "x":
                validDirections.push(["s", "e", "w"]);
                break;
            default:
                return [];
        }
    }
    return getAllValidPermutationsForOneRunner(validDirections, 0, [""]);
}
function getAllValidPermutationsForOneRunner(validDirections, index, partialMovements) {
    if (index >= validDirections.length) {
        return partialMovements;
    }
    const newPartialMovements = [];
    partialMovements.forEach((partialMovement) => {
        validDirections[index].forEach((validDirection) => {
            newPartialMovements.push(partialMovement + validDirection);
        });
    });
    return getAllValidPermutationsForOneRunner(validDirections, index + 1, newPartialMovements);
}
function getMovementsStringFromArray(movementsArray) {
    let movementsString = "";
    for (let i = 0; i < movementsArray.length; i++) {
        if (i > 0) {
            movementsString += "-";
        }
        movementsString += movementsArray[i];
    }
    return movementsString;
}
