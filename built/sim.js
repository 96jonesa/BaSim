'use strict';
import { FoodType } from "./FoodType.js";
import { BarbarianAssault } from "./BarbarianAssault.js";
import { Renderer } from "./Renderer.js";
import { Position } from "./Position.js";
import { LOS_EAST_MASK, LOS_FULL_MASK, LOS_NORTH_MASK, LOS_SOUTH_MASK, LOS_WEST_MASK, MOVE_EAST_MASK, MOVE_FULL_MASK, MOVE_NORTH_MASK, MOVE_SOUTH_MASK, MOVE_WEST_MASK } from "./BarbarianAssaultMap.js";
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
const HTML_TOGGLE_LOG_TO_REPAIR = "toggleloghammertorepair";
const HTML_MARKER_COLOR = "marker";
const HTML_MARKING_TILES = "markingtiles";
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
var toggleMarkingTilesButton;
var savedBarbarianAssault;
var savedTickCountSpanInnerHTML;
var savedCurrentDefenderFoodSpanInnerHTML;
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
    toggleMarkingTilesButton = document.getElementById(HTML_MARKING_TILES);
    toggleMarkingTilesButton.onclick = toggleMarkingTilesButtonOnClick;
    markingTiles = false;
    markedTiles = [];
    reset();
    window.onkeydown = windowOnKeyDown;
    canvas.onmousedown = canvasOnMouseDown;
    canvas.oncontextmenu = function (mouseEvent) {
        mouseEvent.preventDefault();
    };
    infiniteFood = toggleInfiniteFood.value === "yes";
    requireLogs = toggleLogToRepair.value === "yes";
    requireRepairs = toggleRepair.value === "yes";
    wave = Number(waveSelect.value);
    defenderLevel = Number(defenderLevelSelection.value);
    markerColor = Number("0x" + markerColorInput.value.substring(1));
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
    barbarianAssault = new BarbarianAssault(wave, requireRepairs, requireLogs, infiniteFood, [], defenderLevel);
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
                barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);
                break;
            case "c":
                barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);
                break;
            case "w":
                barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.WORMS);
                break;
            case "1":
                barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;
                break;
            case "2":
                barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.CRACKERS;
                break;
            case "3":
                barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.WORMS;
                break;
            case "l":
                barbarianAssault.defenderPlayer.isPickingUpLogs = true;
                break;
            case "r":
                barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);
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
}
/**
 * Pauses and loads the previously saved state of the simulator.
 */
function load() {
    isPaused = true;
    tickCountSpan.innerHTML = savedTickCountSpanInnerHTML;
    currentDefenderFoodSpan.innerHTML = savedCurrentDefenderFoodSpanInnerHTML;
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
                if ((markedTiles[i][0] === xTile) && (markedTiles[i][1] === yTile)) {
                    tileAlreadyMarked = true;
                    markedTiles.splice(i, 1);
                }
            }
            if (!tileAlreadyMarked) {
                markedTiles.push([xTile, yTile]);
            }
            if (!isRunning) {
                draw();
            }
        }
        else {
            barbarianAssault.defenderPlayer.findPath(barbarianAssault, new Position(xTile, yTile));
        }
    }
    else if (mouseEvent.button === 2) {
        if (barbarianAssault.collectorPlayer.position.equals(new Position(xTile, yTile))) {
            barbarianAssault.collectorPlayer.position.x = -1;
        }
        else {
            barbarianAssault.collectorPlayer.position = new Position(xTile, yTile);
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
        renderer.outline(markedTile[0], markedTile[1]);
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
        isRunning = true;
        isPaused = false;
        startStopButton.innerHTML = "Stop Wave";
        barbarianAssault = new BarbarianAssault(wave, requireRepairs, requireLogs, infiniteFood, movements, defenderLevel);
        console.log("Wave " + wave + " started!");
        tick();
        tickTimerId = setInterval(tick, Number(tickDurationInput.value));
    }
}
/**
 * Progresses the state of the simulator by a single tick.
 */
function tick() {
    if (!isPaused) {
        barbarianAssault.tick();
        currentDefenderFoodSpan.innerHTML = barbarianAssault.defenderFoodCall.toString();
        tickCountSpan.innerHTML = barbarianAssault.ticks.toString();
        draw();
    }
}
/**
 * Toggles whether tile-marking mode is enabled.
 */
function toggleMarkingTilesButtonOnClick() {
    markingTiles = !markingTiles;
}
/**
 * Sets the wave to the selected wave value, and stops and resets the simulator.
 */
function waveSelectOnChange() {
    wave = Number(waveSelect.value);
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
    requireRepairs = toggleRepair.value === "yes";
}
/**
 * Toggles whether the simulator must be paused before saving / loading.
 */
function togglePauseSaveLoadOnChange() {
    pauseSaveLoad = togglePauseSaveLoad.value === "yes";
}
/**
 * Toggles whether the defender has infinite food.
 */
function toggleInfiniteFoodOnChange() {
    infiniteFood = toggleInfiniteFood.value === "yes";
}
/**
 * Toggles whether a log is required to repair a trap.
 */
function toggleLogToRepairOnChange() {
    requireLogs = toggleLogToRepair.value === "yes";
}
