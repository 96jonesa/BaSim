'use strict';
import {FoodType} from "./FoodType.js";
import {BarbarianAssault} from "./BarbarianAssault.js";
import {Renderer} from "./Renderer.js";
import {Position} from "./Position.js";
import {
    LOS_EAST_MASK,
    LOS_FULL_MASK, LOS_NORTH_MASK, LOS_SOUTH_MASK,
    LOS_WEST_MASK,
    MOVE_EAST_MASK,
    MOVE_FULL_MASK, MOVE_NORTH_MASK, MOVE_SOUTH_MASK,
    MOVE_WEST_MASK
} from "./BarbarianAssaultMap.js";
import {Food} from "./Food.js";
import {FoodZone} from "./FoodZone.js";
import {RunnerPenance} from "./RunnerPenance.js";
import {HealerPenance} from "./HealerPenance.js";

const HTML_CANVAS: string = "basimcanvas";
const HTML_RUNNER_MOVEMENTS: string = "runnermovements";
const HTML_START_BUTTON: string = "wavestart";
const HTML_WAVE_SELECT: string = "waveselect";
const HTML_TICK_COUNT: string = "tickcount";
const HTML_DEF_LEVEL_SELECT: string = "deflevelselect";
const HTML_TOGGLE_REPAIR: string = 'togglerepair'
const HTML_TOGGLE_PAUSE_SL: string = 'togglepausesl';
const HTML_CURRENT_DEF_FOOD: string = "currdeffood";
const HTML_TICK_DURATION: string = "tickduration";
const HTML_TOGGLE_INFINITE_FOOD: string = "toggleinfinitefood";
const HTML_TOGGLE_LOG_TO_REPAIR: string = "togglelogtorepair";
const HTML_MARKER_COLOR: string = "marker";
const HTML_MARKING_TILES: string = "markingtiles";
const HTML_MAIN_ATTACKER_COMMANDS: string = "mainattackercommands";
const HTML_SECOND_ATTACKER_COMMANDS: string = "secondattackercommands";
const HTML_HEALER_COMMANDS: string = "healercommands";
const HTML_COLLECTOR_COMMANDS: string = "collectorcommands";
const HTML_DEFENDER_COMMANDS: string = "defendercommands";
const HTML_PLAYER_SELECT: string = "playerselect";
const HTML_CONTROLLED_COMMANDS: string = "controlledcommands";

window.onload = init;

var markingTiles: boolean;
var markedTiles: Array<Array<number>>;
var canvas: HTMLCanvasElement;
var movementsInput: HTMLInputElement;
var tickDurationInput: HTMLInputElement;
var startStopButton: HTMLElement;
var waveSelect: HTMLInputElement;
var defenderLevelSelection: HTMLInputElement;
var toggleRepair: HTMLInputElement;
var togglePauseSaveLoad: HTMLInputElement;
var toggleInfiniteFood: HTMLInputElement;
var toggleLogToRepair: HTMLInputElement;
var tickCountSpan: HTMLElement;
var currentDefenderFoodSpan: HTMLElement;
var markerColorInput: HTMLInputElement;
var isRunning: boolean = false;
var barbarianAssault: BarbarianAssault;
var infiniteFood: boolean;
var isPaused: boolean;
var pauseSaveLoad: boolean;
var saveExists: boolean;
var renderer: Renderer;
var requireLogs: boolean;
var requireRepairs: boolean;
var tickTimerId: number;
var wave: number;
var defenderLevel: number;
var markerColor: number;
var toggleMarkingTiles: HTMLInputElement;
var playerSelect: HTMLInputElement;
var player: string;
var controlledCommands: HTMLElement;


var savedBarbarianAssault: BarbarianAssault;
var savedTickCountSpanInnerHTML: string;
var savedCurrentDefenderFoodSpanInnerHTML: string;
var savedPlayer: string;
var savedControlledCommandsInnerHTML: string;
var savedDefenderLevel: string;
var savedWave: string;
var savedMovementsString: string;
var savedMainAttackerCommands: string;
var savedSecondAttackerCommands: string;
var savedHealerCommands: string;
var savedCollectorCommands: string;
var savedDefenderCommands: string;
var savedRequireRepairs: boolean;
var savedInfiniteFood: boolean;
var savedRequireLogs: boolean;

/**
 * Initializes the simulator.
 */
function init(): void {
    canvas = document.getElementById(HTML_CANVAS) as HTMLCanvasElement;
    movementsInput = document.getElementById(HTML_RUNNER_MOVEMENTS) as HTMLInputElement;
    movementsInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    movementsInput.onchange = movementsInputOnChange;
    tickDurationInput = document.getElementById(HTML_TICK_DURATION) as HTMLInputElement;
    startStopButton = document.getElementById(HTML_START_BUTTON);
    startStopButton.onclick = startStopButtonOnClick;
    waveSelect = document.getElementById(HTML_WAVE_SELECT) as HTMLInputElement;
    waveSelect.onchange = waveSelectOnChange;
    defenderLevelSelection = document.getElementById(HTML_DEF_LEVEL_SELECT) as HTMLInputElement;
    defenderLevelSelection.onchange = defenderLevelSelectionOnChange;
    toggleRepair = document.getElementById(HTML_TOGGLE_REPAIR) as HTMLInputElement;
    toggleRepair.onchange = toggleRepairOnChange;
    togglePauseSaveLoad = document.getElementById(HTML_TOGGLE_PAUSE_SL) as HTMLInputElement;
    togglePauseSaveLoad.onchange = togglePauseSaveLoadOnChange;
    toggleInfiniteFood = document.getElementById(HTML_TOGGLE_INFINITE_FOOD) as HTMLInputElement;
    toggleInfiniteFood.onchange = toggleInfiniteFoodOnChange;
    toggleLogToRepair = document.getElementById(HTML_TOGGLE_LOG_TO_REPAIR) as HTMLInputElement;
    toggleLogToRepair.onchange = toggleLogToRepairOnChange;
    tickCountSpan = document.getElementById(HTML_TICK_COUNT);
    currentDefenderFoodSpan = document.getElementById(HTML_CURRENT_DEF_FOOD);
    markerColorInput = document.getElementById(HTML_MARKER_COLOR) as HTMLInputElement;
    renderer = new Renderer(canvas, 64 * 12, 48 * 12, 12);
    toggleMarkingTiles = document.getElementById(HTML_MARKING_TILES) as HTMLInputElement;
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
    canvas.oncontextmenu = function (mouseEvent: MouseEvent): void {
        mouseEvent.preventDefault();
    }
    wave = Number(waveSelect.value);
    defenderLevel = Number(defenderLevelSelection.value);
    markerColor = Number("0x" + markerColorInput.value.substring(1));
    playerSelect = document.getElementById(HTML_PLAYER_SELECT) as HTMLInputElement;
    playerSelect.onchange = playerSelectOnChange;
    player = playerSelect.value;
    controlledCommands = document.getElementById(HTML_CONTROLLED_COMMANDS);
}

/**
 * Resets the simulator: the simulator is stopped and the underlying {@link BarbarianAssault} game
 * is replaced with a new game according to the currently selected configuration.
 */
function reset(): void {
    if (isRunning) {
        clearInterval(tickTimerId);
    }

    isRunning = false;
    startStopButton.innerHTML = "Start Wave";

    console.log((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value);
    console.log(convertCommandsStringToMap((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value));

    barbarianAssault = new BarbarianAssault(
        wave,
        requireRepairs,
        requireLogs,
        infiniteFood,
        [],
        defenderLevel,
        player === "mainattacker" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value),
        player === "secondattacker" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value),
        player === "healer" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value),
        player === "collector" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value),
        player === "defender" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value)
    );

    draw();
}

/**
 * Parses the simulator's configured runner movements, converting them into a list of per-runner
 * movement strings (each formatted e.g. as "wses" to indicate West-South-East-South).
 *
 * @return  a list of per-runner movements strings if the entire runner movements configuration
 *          is valid (i.e. contains only valid characters in the expected format), otherwise null
 */
function parseMovementsInput(): Array<string> {
    const movements: Array<string> = movementsInput.value.split("-");

    for (let i: number = 0; i < movements.length; i++) {
        const moves: string = movements[i];

        for (let j: number = 0; j < moves.length; j++) {
            const move: string = moves[j];

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
function windowOnKeyDown(keyboardEvent: KeyboardEvent): void {
    const key: string = keyboardEvent.key;

    if (isRunning) {
        switch (key) {
            case "t":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);
                }
                break;
            case "c":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);
                }
                break;
            case "w":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.WORMS);
                }
                break;
            case "1":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;
                }
                break;
            case "2":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.CRACKERS;
                }
                break;
            case "3":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.WORMS;
                }
                break;
            case "l":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.isPickingUpLogs = true;
                }
                break;
            case "r":
                if (player === "defender") {
                    barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);
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
function save(): void {
    isPaused = true;

    savedBarbarianAssault = barbarianAssault.clone();
    savedTickCountSpanInnerHTML = tickCountSpan.innerHTML;
    savedCurrentDefenderFoodSpanInnerHTML = currentDefenderFoodSpan.innerHTML;
    savedPlayer = player;
    savedControlledCommandsInnerHTML = controlledCommands.innerHTML;
    savedDefenderLevel = defenderLevelSelection.value;
    savedWave = waveSelect.value;
    savedMovementsString = movementsInput.value;
    savedMainAttackerCommands = (document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value;
    savedSecondAttackerCommands = (document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value;
    savedHealerCommands = (document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value;
    savedCollectorCommands = (document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value;
    savedDefenderCommands = (document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value;
    savedRequireRepairs = requireRepairs;
    savedInfiniteFood = infiniteFood;
    savedRequireLogs = requireLogs;
}

/**
 * Pauses and loads the previously saved state of the simulator.
 */
function load(): void {
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
    (document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value = savedMainAttackerCommands;
    (document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value = savedSecondAttackerCommands;
    (document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value = savedHealerCommands;
    (document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value = savedCollectorCommands;
    (document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value = savedDefenderCommands;
    toggleLogToRepair.checked = savedRequireLogs;
    toggleRepair.checked = savedRequireRepairs;
    toggleInfiniteFood.checked = savedInfiniteFood;

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
function canvasOnMouseDown(mouseEvent: MouseEvent): void {
    const canvasRect: DOMRect = renderer.canvas.getBoundingClientRect();
    const xTile: number = Math.trunc((mouseEvent.clientX - canvasRect.left) / renderer.tileSize);
    const yTile: number = Math.trunc((canvasRect.bottom - 1 - mouseEvent.clientY) / renderer.tileSize);

    if (mouseEvent.button === 0) {
        if (markingTiles) {
            let tileAlreadyMarked: boolean = false;

            for (let i: number = 0; i < markedTiles.length; i++) {
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
        } else {
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
function draw(): void {
    drawMap()
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
function drawMap(): void {
    renderer.setDrawColor(206, 183, 117, 255);
    renderer.clear();

    for (let y: number = 0; y < barbarianAssault.map.height; y++) {
        for (let x: number = 0; x < barbarianAssault.map.width; x++) {
            const flag: number = barbarianAssault.map.getFlag(new Position(x, y));

            if ((flag & LOS_FULL_MASK) !== 0) {
                renderer.setDrawColor(0, 0, 0, 255);
                renderer.fillOpaque(x, y);
            } else {
                if ((flag & MOVE_FULL_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.fillOpaque(x, y);
                }

                if ((flag & LOS_EAST_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.eastLine(x, y);
                } else if ((flag & MOVE_EAST_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.eastLine(x, y);
                }

                if ((flag & LOS_WEST_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.westLine(x, y);
                } else if ((flag & MOVE_WEST_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.westLine(x, y);
                }

                if ((flag & LOS_NORTH_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.northLine(x, y);
                } else if ((flag & MOVE_NORTH_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.northLine(x, y);
                }

                if ((flag & LOS_SOUTH_MASK) !== 0) {
                    renderer.setDrawColor(0, 0, 0, 255);
                    renderer.southLine(x, y);
                } else if ((flag & MOVE_SOUTH_MASK) !== 0) {
                    renderer.setDrawColor(127, 127, 127, 255);
                    renderer.southLine(x, y);
                }
            }
        }
    }

    renderer.setDrawColor((markerColor >> 16) & 255, (markerColor >> 8) & 255, markerColor & 255, 255);

    for (let i: number = 0; i < markedTiles.length; i++) {
        const markedTile: Array<number> = markedTiles[i];
        renderer.outline(markedTile[0], markedTile[1]);
    }
}

/**
 * Draws details of the game (aesthetic details, logs, and traps).
 */
function drawDetails(): void {
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
        renderer.setDrawColor(255, 0, 0 , 255);
    } else if (barbarianAssault.eastTrapCharges === 1) {
        renderer.setDrawColor(255, 140, 0, 255);
    }

    renderer.fill(45, 26);

    renderer.setDrawColor(160, 82, 45, 255);

    if (barbarianAssault.westTrapCharges < 1) {
        renderer.setDrawColor(255, 0, 0 , 255);
    } else if (barbarianAssault.westTrapCharges === 1) {
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
function drawItems(): void {
    for (let i: number = 0; i < barbarianAssault.map.foodZones.length; i++) {
        const foodZone: FoodZone = barbarianAssault.map.foodZones[i];

        for (let j: number = 0; j < foodZone.foodList.length; j++) {
            const food: Food = foodZone.foodList[j];
            renderer.setDrawColor(food.colorRed, food.colorGreen, food.colorBlue, 127);
            renderer.fillItem(food.position.x, food.position.y);
        }
    }
}

/**
 * Draws entities ({@link Character}s}.
 */
function drawEntities(): void {
    renderer.setDrawColor(10, 10, 240, 127);

    barbarianAssault.runners.forEach((runner: RunnerPenance): void => {
        renderer.fill(runner.position.x, runner.position.y);
    });

    renderer.setDrawColor(10, 240, 10, 127);

    barbarianAssault.healers.forEach((healer: HealerPenance): void => {
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
function drawGrid(): void {
    for (let xTile: number = 0; xTile < barbarianAssault.map.width; xTile++) {
        if (xTile % 8 === 7) {
            renderer.setDrawColor(0, 0, 0, 72);
        } else {
            renderer.setDrawColor(0, 0, 0, 48);
        }

        renderer.eastLineBig(xTile, 0, barbarianAssault.map.height);
    }

    for (let yTile: number = 0; yTile < barbarianAssault.map.height; yTile++) {
        if (yTile % 8 === 7) {
            renderer.setDrawColor(0, 0, 0, 72);
        } else {
            renderer.setDrawColor(0, 0, 0, 48);
        }

        renderer.northLineBig(0, yTile, barbarianAssault.map.width);
    }
}

/**
 * Draws aesthetic overlays.
 */
function drawOverlays(): void {
    renderer.setDrawColor(240, 10, 10, 220);

    if (wave === 10) {
        renderer.outline(18, 38);
    } else {
        renderer.outline(18, 37);
    }

    renderer.outline(24, 39);
    renderer.fill(33, 6);

    renderer.setDrawColor(10, 10, 240, 220);

    if (wave === 10) {
        renderer.outline(42, 38);
    } else {
        renderer.outline(36, 39);
    }

    renderer.fill(34, 6);

    renderer.setDrawColor(10, 240, 10, 220);

    if (wave === 10) {
        renderer.outline(36, 39);
    } else {
        renderer.outline(42, 37);
    }

    renderer.fill(35, 6);

    renderer.setDrawColor(240, 240, 10, 220);;
    renderer.fill(36, 6);
}

/**
 * If the simulator is running, then stops and resets the simulator.
 * Otherwise, starts the simulator.
 */
function startStopButtonOnClick(): void {
    if (isRunning) {
        barbarianAssault.map.reset();
        reset();
    } else {
        const movements: Array<string> = parseMovementsInput();

        if (movements === null) {
            alert("Invalid runner movements. Example: ws-s");
            return;
        }

        isRunning = true;
        isPaused = false;
        startStopButton.innerHTML = "Stop Wave";

        controlledCommands.innerHTML = "";

        barbarianAssault = new BarbarianAssault(
            wave,
            requireRepairs,
            requireLogs,
            infiniteFood,
            movements,
            defenderLevel,
            player === "mainattacker" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value),
            player === "secondattacker" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value),
            player === "healer" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value),
            player === "collector" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value),
            player === "defender" ? new Map<number, Position> : convertCommandsStringToMap((document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value)
        );

        console.log("Wave " + wave + " started!");
        tick();
        tickTimerId = setInterval(tick, Number(tickDurationInput.value));
    }
}

/**
 * Progresses the state of the simulator by a single tick.
 */
function tick(): void {
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
function toggleMarkingTilesOnChange(): void {
    markingTiles = toggleMarkingTiles.checked;
}

/**
 * Sets the wave to the selected wave value, and stops and resets the simulator.
 */
function waveSelectOnChange(): void {
    wave = Number(waveSelect.value);
    reset();
}

function playerSelectOnChange(): void {
    player = playerSelect.value;
    reset();
}

/**
 * Sets the defender level to the selected defender level value, and stops and resets the simulator.
 */
function defenderLevelSelectionOnChange(): void {
    defenderLevel = Number(defenderLevelSelection.value);
    reset();
}

/**
 * Toggles whether traps need to be repaired.
 */
function toggleRepairOnChange(): void {
    requireRepairs = toggleRepair.checked;
    reset();
}

function movementsInputOnChange(): void {
    reset();
}

/**
 * Toggles whether the simulator must be paused before saving / loading.
 */
function togglePauseSaveLoadOnChange(): void {
    pauseSaveLoad = togglePauseSaveLoad.checked;
}

/**
 * Toggles whether the defender has infinite food.
 */
function toggleInfiniteFoodOnChange(): void {
    infiniteFood = toggleInfiniteFood.checked;
    reset();
}

/**
 * Toggles whether a log is required to repair a trap.
 */
function toggleLogToRepairOnChange(): void {
    requireLogs = toggleLogToRepair.checked;
    reset();
}

/**
 * Converts the given commands string to a map from tick numbers to positions.
 * If the given commands string is invalid, then an empty map is returned
 *
 * @param commandsString    the commands string to convert to a map from tick
 *                          numbers to positions
 * @return                  a map from tick numbers to positions as specified by
 *                          the given commands string, or an empty map if the given
 *                          commands string is invalid
 */
function convertCommandsStringToMap(commandsString: string): Map<number, Position> {
    if (commandsString === null) {
        return new Map<number, Position>();
    }

    const commandsMap: Map<number, Position> = new Map<number, Position>();

    const commands: Array<string> = commandsString.split("\n");

    for (let i: number = 0; i < commands.length; i++) {
        const command: string = commands[i];

        if (command.length === 0) {
            continue;
        }

        const commandTokens: Array<string> = command.split(":");

        if (commandTokens.length !== 2) {
            return new Map<number, Position>();
        }

        const tick: number = Number(commandTokens[0]);

        if (!Number.isInteger(tick) || tick < 1) {
            return new Map<number, Position>();
        }

        const positionTokens: Array<string> = commandTokens[1].split(",");
        const positionX: number = Number(positionTokens[0]);
        const positionY: number = Number(positionTokens[1]);

        if (!Number.isInteger(positionX) || !Number.isInteger(positionY)) {
            return new Map<number, Position>();
        }

        commandsMap.set(tick, new Position(positionX, positionY));
    }

    return commandsMap;
}