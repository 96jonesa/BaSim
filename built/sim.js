'use strict';
import { FoodType } from "./FoodType.js";
import { BarbarianAssault } from "./BarbarianAssault.js";
import { Renderer } from "./Renderer.js";
import { Position } from "./Position.js";
import { LOS_EAST_MASK, LOS_FULL_MASK, LOS_NORTH_MASK, LOS_SOUTH_MASK, LOS_WEST_MASK, MOVE_EAST_MASK, MOVE_FULL_MASK, MOVE_NORTH_MASK, MOVE_SOUTH_MASK, MOVE_WEST_MASK } from "./BarbarianAssaultMap.js";
import { MoveCommand } from "./MoveCommand.js";
import { DefenderActionCommand } from "./DefenderActionCommand.js";
import { DefenderActionType } from "./DefenderActionType.js";
import { TileMarker } from "./TileMarker.js";
import { RGBColor } from "./RGBColor.js";
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
var toggleMarkingTiles;
var playerSelect;
var player;
var controlledCommands;
var foodCallsInput;
var runnerMovementsToCheckInput;
var runnersDeadByTickInput;
var simulateButton;
var runnersDoNotDieWithMovements;
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
    tickCountSpan = document.getElementById(HTML_TICK_COUNT);
    currentDefenderFoodSpan = document.getElementById(HTML_CURRENT_DEF_FOOD);
    markerColorInput = document.getElementById(HTML_MARKER_COLOR);
    renderer = new Renderer(canvas, 64 * 12, 48 * 12, 12);
    toggleMarkingTiles = document.getElementById(HTML_MARKING_TILES);
    toggleMarkingTiles.onchange = toggleMarkingTilesOnChange;
    markingTiles = toggleMarkingTiles.checked;
    markedTiles = [];
    infiniteFood = toggleInfiniteFood.checked;
    requireRepairs = toggleRepair.checked;
    requireLogs = toggleLogToRepair.checked;
    pauseSaveLoad = togglePauseSaveLoad.checked;
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
/**
 * Handles the given keyboard event.
 *
 * @param keyboardEvent the keyboard event to handle
 */
function windowOnKeyDown(keyboardEvent) {
    const key = keyboardEvent.key;
    if (isRunning) {
        switch (key) {
            case "t":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":t<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "c":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":c<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "w":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.WORMS);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":w<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "1":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":1<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "2":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.CRACKERS;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":2<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "3":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.WORMS;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":3<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "l":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.isPickingUpLogs = true;
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":l<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "r":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);
                    controlledCommands.innerHTML += barbarianAssault.ticks + ":r<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "p":
                isPaused = !isPaused;
                break;
            case "s":
                if (isPaused || !pauseSaveLoad) {
                    isPaused = true;
                    save();
                    saveExists = true;
                }
                break;
            case "y":
                if (saveExists && (isPaused || !pauseSaveLoad)) {
                    isPaused = true;
                    load();
                }
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
    barbarianAssault = savedBarbarianAssault;
    // the existing save state will mutate as the simulator proceeds,
    // so re-clone the save state in case of subsequent loads
    save();
    draw();
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
    renderer.setDrawColor(10, 10, 240, 127);
    barbarianAssault.runners.forEach((runner) => {
        renderer.fill(runner.position.x, runner.position.y);
    });
    renderer.setDrawColor(10, 240, 10, 127);
    barbarianAssault.healers.forEach((healer) => {
        renderer.fill(healer.position.x, healer.position.y);
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
            renderer.setDrawColor(0, 0, 0, 72);
        }
        else {
            renderer.setDrawColor(0, 0, 0, 48);
        }
        renderer.eastLineBig(xTile, 0, barbarianAssault.map.height);
    }
    for (let yTile = 0; yTile < barbarianAssault.map.height; yTile++) {
        if (yTile % 8 === 7) {
            renderer.setDrawColor(0, 0, 0, 72);
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
        const foodCalls = parseFoodCallsInput();
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
        isRunning = true;
        isPaused = false;
        startStopButton.innerHTML = "Stop Wave";
        controlledCommands.innerHTML = "";
        barbarianAssault = new BarbarianAssault(wave, requireRepairs, requireLogs, infiniteFood, movements, defenderLevel, player === "mainattacker" ? new Map : mainAttackerCommands, player === "secondattacker" ? new Map : secondAttackerCommands, player === "healer" ? new Map : healerCommands, player === "collector" ? new Map : collectorCommands, player === "defender" ? new Map : defenderCommands, foodCalls);
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
function tick() {
    if (!isPaused) {
        barbarianAssault.tick();
        currentDefenderFoodSpan.innerHTML = barbarianAssault.defenderFoodCall.toString();
        tickCountSpan.innerHTML = barbarianAssault.ticks.toString() + " (" + ticksToSeconds(barbarianAssault.ticks) + "s)";
        draw();
    }
}
/**
 * Toggles whether tile-marking mode is enabled.
 */
function toggleMarkingTilesOnChange() {
    markingTiles = toggleMarkingTiles.checked;
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
    const barbarianAssaultSim = new BarbarianAssault(wave, requireRepairs, requireLogs, infiniteFood, runnerMovements, defenderLevel, mainAttackerCommands, secondAttackerCommands, healerCommands, collectorCommands, defenderCommands, foodCalls);
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
