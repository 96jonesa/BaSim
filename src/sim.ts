'use strict';
import {FoodType} from "./FoodType.js";
import {BarbarianAssault} from "./BarbarianAssault.js";
import {Renderer} from "./Renderer.js";
import {Position} from "./Position.js";
import {
    LOS_EAST_MASK,
    LOS_FULL_MASK,
    LOS_NORTH_MASK,
    LOS_SOUTH_MASK,
    LOS_WEST_MASK,
    MOVE_EAST_MASK,
    MOVE_FULL_MASK,
    MOVE_NORTH_MASK,
    MOVE_SOUTH_MASK,
    MOVE_WEST_MASK
} from "./BarbarianAssaultMap.js";
import {Food} from "./Food.js";
import {FoodZone} from "./FoodZone.js";
import {RunnerPenance} from "./RunnerPenance.js";
import {HealerPenance} from "./HealerPenance.js";
import {Command} from "./Command.js";
import {MoveCommand} from "./MoveCommand.js";
import {DefenderActionCommand} from "./DefenderActionCommand.js";
import {DefenderActionType} from "./DefenderActionType.js";
import {TileMarker} from "./TileMarker.js";
import {RGBColor} from "./RGBColor.js";
import {Player} from "./Player.js";
import {HealerTargetType} from "./HealerTargetType.js";
import {parseCannonInput, getCannonPosition} from "./Cannon.js";
import {CannonCommand} from "./CannonCommand.js";
import {CannonSide} from "./CannonSide.js";
import {EggType} from "./EggType.js";
import {HealerCodeCommand} from "./HealerCodeCommand.js";
import {HealerCodeAction} from "./HealerCodeAction.js";
import {WalkRunCommand} from "./WalkRunCommand.js";
import {ToggleRunCommand} from "./ToggleRunCommand.js";
import {SeedCommand} from "./SeedCommand.js";
import {RedXCommand} from "./RedXCommand.js";
import {RedXMoveCommand} from "./RedXMoveCommand.js";
import {DefenderPickupAtCommand} from "./DefenderPickupAtCommand.js";

const HTML_CANVAS: string = "basimcanvas";
const HTML_RUNNER_MOVEMENTS: string = "runnermovements";
const HTML_START_BUTTON: string = "wavestart";
const HTML_WAVE_SELECT: string = "waveselect";
const HTML_TICK_COUNT: string = "tickcount";
const HTML_DEF_LEVEL_SELECT: string = "deflevelselect";
const HTML_TOGGLE_REPAIR: string = 'togglerepair'
const HTML_TOGGLE_PAUSE_SL: string = 'togglepausesl';
const HTML_TOGGLE_SEED_QUEUE_PATH: string = 'toggleseedqueuepath';
const HTML_CURRENT_DEF_FOOD: string = "currdeffood";
const HTML_TICK_DURATION: string = "tickduration";
const HTML_TOGGLE_INFINITE_FOOD: string = "toggleinfinitefood";
const HTML_TOGGLE_LOG_TO_REPAIR: string = "togglelogtorepair";
const HTML_TOGGLE_IGNORE_MAX_HEALERS: string = "toggleignoremaxhealers";
const HTML_MARKER_COLOR: string = "marker";
const HTML_MARKING_TILES: string = "markingtiles";
const HTML_MAIN_ATTACKER_COMMANDS: string = "mainattackercommands";
const HTML_SECOND_ATTACKER_COMMANDS: string = "secondattackercommands";
const HTML_HEALER_COMMANDS: string = "healercommands";
const HTML_COLLECTOR_COMMANDS: string = "collectorcommands";
const HTML_DEFENDER_COMMANDS: string = "defendercommands";
const HTML_PLAYER_SELECT: string = "playerselect";
const HTML_CONTROLLED_COMMANDS: string = "controlledcommands";
const HTML_FOOD_CALLS: string = "foodcalls";
const HTML_RUNNER_MOVEMENTS_TO_CHECK: string = "runnermovementstocheck";
const HTML_RUNNERS_DEAD_BY_TICK: string = "runnersdeadbytick";
const HTML_SIMULATE: string = "simulate";
const HTML_RUNNERS_DO_NOT_DIE_WITH_MOVEMENTS: string = "runnersdonotdiemovements";
const HTML_CANNON_QUEUE: string = "cannonqueue";
const HTML_RUNNER_TABLE: string = "runnertable";
const HTML_HEALER_TABLE: string = "healertable";
const HTML_TOGGLE_DARK_MODE: string = "toggledarkmode";
const HTML_TOGGLE_RENDER_DISTANCE: string = "togglerenderdistance";
const HTML_TOGGLE_SIMPLE_FOOD: string = "togglesimplefood";
const HTML_FOOD_CALLS_ROW: string = "foodcallsrow";
const HTML_INFINITE_FOOD_ROW: string = "infinitefoodrow";
const HTML_CURRENT_FOOD_ROW: string = "currfoodrow";
const HTML_HEALER_SPAWN_TARGETS: string = "healerspawntargets";
const HTML_RUNNER_SPAWNS: string = "runnerspawns";
const HTML_HEALER_SPAWNS: string = "healerspawns";
const HTML_COPY_CONTROLLED_COMMANDS: string = "copycontrolledcommands";
const HTML_EXPORT_MARKERS: string = "exportmarkers";
const HTML_IMPORT_MARKERS: string = "importmarkers";
const HTML_CLEAR_MARKERS: string = "clearmarkers";
const HTML_MARKER_IMPORT_FIELD: string = "markerimportfield";
const HTML_SETTINGS_EXPORT: string = "settingsexport";
const HTML_SETTINGS_IMPORT: string = "settingsimport";
const HTML_SETTINGS_IMPORT_FIELD: string = "settingsimportfield";
const HTML_TOGGLE_SECONDS: string = "toggleseconds";
const HTML_START_TICK: string = "starttick";
const HTML_PAUSE_RESUME: string = "pauseresume";
const HTML_STEP_BACK: string = "stepback";
const HTML_STEP_FORWARD: string = "stepforward";
const HTML_SAVE_STATE: string = "savestate";
const HTML_LOAD_STATE: string = "loadstate";

window.onload = init;

var markingTiles: boolean;
var markedTiles: Array<TileMarker>;

const TILE_MARKERS_STORAGE_KEY: string = "mclovin-ba-sim-tile-markers";

function saveMarkedTiles(): void {
    const data = markedTiles.map((t: TileMarker) => ({
        x: t.position.x,
        y: t.position.y,
        r: t.rgbColor.r,
        g: t.rgbColor.g,
        b: t.rgbColor.b,
    }));
    localStorage.setItem(TILE_MARKERS_STORAGE_KEY, JSON.stringify(data));
}

function loadMarkedTiles(): Array<TileMarker> {
    const json = localStorage.getItem(TILE_MARKERS_STORAGE_KEY);
    if (json === null) {
        return [];
    }
    try {
        const data = JSON.parse(json);
        return data.map((t: {x: number, y: number, r: number, g: number, b: number}) =>
            new TileMarker(new Position(t.x, t.y), new RGBColor(t.r, t.g, t.b))
        );
    } catch (e) {
        return [];
    }
}

var eggImages: Record<string, HTMLImageElement> = {};

var canvas: HTMLCanvasElement;
var movementsInput: HTMLInputElement;
var tickDurationInput: HTMLInputElement;
var startStopButton: HTMLButtonElement;
var waveSelect: HTMLInputElement;
var defenderLevelSelection: HTMLInputElement;
var toggleRepair: HTMLInputElement;
var togglePauseSaveLoad: HTMLInputElement;
var toggleInfiniteFood: HTMLInputElement;
var toggleLogToRepair: HTMLInputElement;
var toggleSeedQueuePath: HTMLInputElement;
var tickCountSpan: HTMLElement;
var currentDefenderFoodSpan: HTMLElement;
var markerColorInput: HTMLInputElement;
var isRunning: boolean = false;
var barbarianAssault: BarbarianAssault;
var lastClickTick: number = -1;
var infiniteFood: boolean;
var simpleFood: boolean = true;
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
var toggleRenderDistance: HTMLInputElement;
var toggleIgnoreMaxHealers: HTMLInputElement;
var toggleMarkingTiles: HTMLInputElement;
var playerSelect: HTMLInputElement;
var player: string;
var controlledCommands: HTMLElement;
var foodCallsInput: HTMLInputElement;
var runnerMovementsToCheckInput: HTMLTextAreaElement;
var runnersDeadByTickInput: HTMLInputElement;
var simulateButton: HTMLButtonElement;
var runnersDoNotDieWithMovements: HTMLElement;
var cannonQueueInput: HTMLInputElement;
var healerSpawnTargetsInput: HTMLInputElement;
var runnerSpawnsInput: HTMLInputElement;
var healerSpawnsInput: HTMLInputElement;
var toggleSecondsButton: HTMLButtonElement;
var secondsMode: boolean = false;
var startTickInput: HTMLInputElement;
var pauseResumeButton: HTMLButtonElement;
var stepBackButton: HTMLButtonElement;
var stepForwardButton: HTMLButtonElement;
var saveStateButton: HTMLButtonElement;
var loadStateButton: HTMLButtonElement;

const STATE_HISTORY_LIMIT: number = 1000;
var stateHistory: Array<{ba: BarbarianAssault, tickHTML: string, foodHTML: string, commandsHTML: string}> = [];
var stateIndex: number = -1;

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
var savedFoodCallsString: string;
var savedCannonQueueString: string;
var savedHealerSpawnTargetsString: string;
var savedRunnerSpawnsString: string;
var savedHealerSpawnsString: string;

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
    foodCallsInput = document.getElementById(HTML_FOOD_CALLS) as HTMLInputElement;
    foodCallsInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    foodCallsInput.onchange = foodCallsInputOnChange;
    tickDurationInput = document.getElementById(HTML_TICK_DURATION) as HTMLInputElement;
    startStopButton = document.getElementById(HTML_START_BUTTON) as HTMLButtonElement;
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
    toggleRenderDistance = document.getElementById(HTML_TOGGLE_RENDER_DISTANCE) as HTMLInputElement;
    toggleIgnoreMaxHealers = document.getElementById(HTML_TOGGLE_IGNORE_MAX_HEALERS) as HTMLInputElement;
    toggleSeedQueuePath = document.getElementById(HTML_TOGGLE_SEED_QUEUE_PATH) as HTMLInputElement;
    tickCountSpan = document.getElementById(HTML_TICK_COUNT);
    currentDefenderFoodSpan = document.getElementById(HTML_CURRENT_DEF_FOOD);
    markerColorInput = document.getElementById(HTML_MARKER_COLOR) as HTMLInputElement;
    renderer = new Renderer(canvas, 48 * 12, 48 * 12, 12, 8);
    for (const type of ["r", "g", "b"]) {
        const img = new Image();
        img.src = "static/" + (type === "r" ? "red" : type === "g" ? "green" : "blue") + "_egg.svg";
        eggImages[type] = img;
    }
    toggleMarkingTiles = document.getElementById(HTML_MARKING_TILES) as HTMLInputElement;
    toggleMarkingTiles.onchange = toggleMarkingTilesOnChange;
    markingTiles = toggleMarkingTiles.checked;
    markedTiles = loadMarkedTiles();
    infiniteFood = toggleInfiniteFood.checked;

    const toggleSimpleFoodButton = document.getElementById(HTML_TOGGLE_SIMPLE_FOOD) as HTMLButtonElement;
    toggleSimpleFoodButton.onclick = function (): void {
        simpleFood = !simpleFood;
        toggleSimpleFoodButton.innerHTML = simpleFood ? "Disable simple food" : "Enable simple food";
        const display = simpleFood ? "none" : "";
        document.getElementById(HTML_FOOD_CALLS_ROW).style.display = display;
        document.getElementById(HTML_INFINITE_FOOD_ROW).style.display = display;
        document.getElementById(HTML_CURRENT_FOOD_ROW).style.display = display;
        document.getElementById("hotkeylegend-normal").style.display = display;
        document.getElementById("hotkeylegend-simple").style.display = simpleFood ? "" : "none";
        document.getElementById("tclegend-normal").style.display = display;
        document.getElementById("tclegend-simple").style.display = simpleFood ? "" : "none";
        const tip = document.getElementById("teamcommandstip");
        const baseTip = "Enter commands as tick:x,y per line to move a player to (x,y) at that tick. e.g. 5:20,21. Multiple commands per tick allowed.\n\n";
        const defTip = simpleFood
            ? "Defender actions: tick:r/w/e/l/t performs that key action at the specified tick. tick:e,x,y picks up any food at (x,y).\n\n"
            : "Defender actions: tick:t/c/w/u/i/o/e/l/r performs that key action at the specified tick. tick:e,x,y picks up any food at (x,y).\n\n";
        const healerTip = "Healer codes: tick:h<id>,<count> for any player. e.g. 1:h1,3 poisons healer 1 three times starting at its spawn tick. Player auto-pathfinds and uses poison food.\n\n";
        const redXTip = "Red x: tick:x1-x8 sets red x on healer. tick:>x,y sets red x path.\n\n";
        const walkTip = "Walk/run: tick:m toggles, tick:walk/tick:run sets explicitly.";
        tip.title = baseTip + defTip + healerTip + redXTip + walkTip;
        convertDefenderCommands(simpleFood);
        (document.getElementById("settingsexportsolar") as HTMLButtonElement).disabled = !simpleFood;
    };

    toggleSecondsButton = document.getElementById(HTML_TOGGLE_SECONDS) as HTMLButtonElement;
    toggleSecondsButton.onclick = toggleSecondsOnClick;
    startTickInput = document.getElementById(HTML_START_TICK) as HTMLInputElement;
    startTickInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };

    requireRepairs = toggleRepair.checked;
    requireLogs = toggleLogToRepair.checked;
    pauseSaveLoad = togglePauseSaveLoad.checked;
    pauseResumeButton = document.getElementById(HTML_PAUSE_RESUME) as HTMLButtonElement;
    stepBackButton = document.getElementById(HTML_STEP_BACK) as HTMLButtonElement;
    stepForwardButton = document.getElementById(HTML_STEP_FORWARD) as HTMLButtonElement;
    saveStateButton = document.getElementById(HTML_SAVE_STATE) as HTMLButtonElement;
    loadStateButton = document.getElementById(HTML_LOAD_STATE) as HTMLButtonElement;
    reset();
    window.onkeydown = windowOnKeyDown;
    canvas.onmousedown = canvasOnMouseDown;
    canvas.oncontextmenu = function (mouseEvent: MouseEvent): void {
        mouseEvent.preventDefault();
    }
    wave = Number(waveSelect.value);
    defenderLevel = Number(defenderLevelSelection.value);
    markerColor = Number("0x" + markerColorInput.value.substring(1));
    markerColorInput.onchange = markerColorInputOnChange;
    playerSelect = document.getElementById(HTML_PLAYER_SELECT) as HTMLInputElement;
    playerSelect.onchange = playerSelectOnChange;
    player = playerSelect.value;
    controlledCommands = document.getElementById(HTML_CONTROLLED_COMMANDS);
    runnerMovementsToCheckInput = document.getElementById(HTML_RUNNER_MOVEMENTS_TO_CHECK) as HTMLTextAreaElement;
    runnerMovementsToCheckInput.onchange = runnerMovementsToCheckInputOnChange;
    cannonQueueInput = document.getElementById(HTML_CANNON_QUEUE) as HTMLInputElement;
    cannonQueueInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    healerSpawnTargetsInput = document.getElementById(HTML_HEALER_SPAWN_TARGETS) as HTMLInputElement;
    healerSpawnTargetsInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnerSpawnsInput = document.getElementById(HTML_RUNNER_SPAWNS) as HTMLInputElement;
    runnerSpawnsInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    healerSpawnsInput = document.getElementById(HTML_HEALER_SPAWNS) as HTMLInputElement;
    healerSpawnsInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnersDeadByTickInput = document.getElementById(HTML_RUNNERS_DEAD_BY_TICK) as HTMLInputElement;
    runnersDeadByTickInput.onkeydown = function (keyboardEvent: KeyboardEvent): void {
        if (keyboardEvent.key === " ") {
            keyboardEvent.preventDefault();
        }
    };
    runnersDeadByTickInput.onchange = runnersDeadByTickInputOnChange;
    simulateButton = document.getElementById(HTML_SIMULATE) as HTMLButtonElement;
    simulateButton.onclick = simulateButtonOnClick;
    runnersDoNotDieWithMovements = document.getElementById(HTML_RUNNERS_DO_NOT_DIE_WITH_MOVEMENTS);

    const darkModeToggle = document.getElementById(HTML_TOGGLE_DARK_MODE) as HTMLInputElement;
    const savedDarkMode = localStorage.getItem("mclovin-ba-sim-dark-mode");
    if (savedDarkMode === "true") {
        document.body.classList.add("dark");
        darkModeToggle.checked = true;
    }
    darkModeToggle.onchange = function (): void {
        document.body.classList.toggle("dark", darkModeToggle.checked);
        localStorage.setItem("mclovin-ba-sim-dark-mode", String(darkModeToggle.checked));
    };

    (document.getElementById(HTML_COPY_CONTROLLED_COMMANDS) as HTMLButtonElement).onclick = function () {
        const text = (document.getElementById(HTML_CONTROLLED_COMMANDS) as HTMLDivElement).innerText;
        navigator.clipboard.writeText(text);
    };
    (document.getElementById(HTML_EXPORT_MARKERS) as HTMLButtonElement).onclick = exportMarkers;
    (document.getElementById(HTML_IMPORT_MARKERS) as HTMLButtonElement).onclick = importMarkers;
    (document.getElementById(HTML_CLEAR_MARKERS) as HTMLButtonElement).onclick = clearMarkers;
    (document.getElementById(HTML_SETTINGS_EXPORT) as HTMLButtonElement).onclick = exportSettings;
    (document.getElementById(HTML_SETTINGS_IMPORT) as HTMLButtonElement).onclick = importSettings;
    (document.getElementById("settingsimportsolar") as HTMLButtonElement).onclick = importSettingsFromSolar;

    pauseResumeButton.onclick = function (): void {
        if (!isRunning) return;
        isPaused = !isPaused;
        pauseResumeButton.innerHTML = isPaused ? "Resume" : "Pause";
    };
    stepBackButton.onclick = function (): void {
        if (!isRunning) return;
        isPaused = true;
        pauseResumeButton.innerHTML = "Resume";
        stepBackward();
    };
    stepForwardButton.onclick = function (): void {
        if (!isRunning) return;
        isPaused = true;
        pauseResumeButton.innerHTML = "Resume";
        stepForward();
    };
    saveStateButton.onclick = function (): void {
        if (isPaused || !pauseSaveLoad) {
            isPaused = true;
            pauseResumeButton.innerHTML = "Resume";
            save();
            saveExists = true;
        }
    };
    loadStateButton.onclick = function (): void {
        if (!saveExists) return;
        if (!isRunning) {
            isRunning = true;
            startStopButton.innerHTML = "Stop Wave";
            pauseResumeButton.style.display = "";
            stepBackButton.style.display = "";
            stepForwardButton.style.display = "";
            saveStateButton.disabled = false;
            (document.getElementById(HTML_RUNNER_TABLE) as HTMLElement).style.display = "table";
            (document.getElementById(HTML_HEALER_TABLE) as HTMLElement).style.display = "table";
            tickTimerId = setInterval(tick, Number(tickDurationInput.value));
        }
        isPaused = true;
        pauseResumeButton.innerHTML = "Resume";
        load();
    };
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
    pauseResumeButton.style.display = "none";
    stepBackButton.style.display = "none";
    stepForwardButton.style.display = "none";
    saveStateButton.disabled = true;
    (document.getElementById(HTML_RUNNER_TABLE) as HTMLElement).style.display = "none";
    (document.getElementById(HTML_HEALER_TABLE) as HTMLElement).style.display = "none";
    stateHistory = [];
    stateIndex = -1;

    barbarianAssault = new BarbarianAssault(
        wave,
        requireRepairs,
        requireLogs,
        infiniteFood,
        [],
        defenderLevel,
        player === "mainattacker" ? new Map<number, Array<Command>> : convertCommandsStringToMap((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value, "mainattacker"),
        player === "secondattacker" ? new Map<number, Array<Command>> : convertCommandsStringToMap((document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value, "secondattacker"),
        player === "healer" ? new Map<number, Array<Command>> : convertCommandsStringToMap((document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value, "healer"),
        player === "collector" ? new Map<number, Array<Command>> : convertCommandsStringToMap((document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value, "collector"),
        player === "defender" ? new Map<number, Array<Command>> : convertCommandsStringToMap((document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value, "defender"),
        []
    );
    barbarianAssault.ignoreMaxHealers = toggleIgnoreMaxHealers.checked;

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

function isValidMovementPattern(pattern: string): boolean {
    let i: number = 0;
    while (i < pattern.length) {
        const c: string = pattern[i];
        if (c === "[") {
            i++;
            if (i >= pattern.length) return false;
            while (i < pattern.length && pattern[i] !== "]") {
                if (pattern[i] !== "s" && pattern[i] !== "w" && pattern[i] !== "e") return false;
                i++;
            }
            if (i >= pattern.length) return false;
            i++;
        } else if (c === "s" || c === "w" || c === "e" || c === "x") {
            i++;
        } else if (c === "") {
            i++;
        } else {
            return false;
        }
    }
    return true;
}

function parseRunnerMovementsToCheck(): Array<Array<string>> {
    const lines: Array<string> = runnerMovementsToCheckInput.value.split("\n").filter(line => line.trim() !== "");

    if (lines.length === 0) {
        return null;
    }

    const result: Array<Array<string>> = [];

    for (const line of lines) {
        const runnerMovements: Array<string> = line.trim().split("-");

        for (let i: number = 0; i < runnerMovements.length; i++) {
            if (!isValidMovementPattern(runnerMovements[i])) {
                return null;
            }
        }

        result.push(runnerMovements);
    }

    return result;
}

function parseFoodCallsInput(): Array<FoodType> {
    const foodCalls: Array<FoodType> = [];

    const foodCallsString: string = foodCallsInput.value;

    for (let i: number = 0; i < foodCallsString.length; i++) {
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

function parseSpawnsInput(value: string): Array<number> {
    const trimmed = value.trim();
    if (trimmed.length === 0) {
        return [];
    }
    const parts = trimmed.split(/[,\-]/);
    const spawns: Array<number> = [];
    for (const part of parts) {
        if (useSeconds()) {
            const seconds = parseFloat(part.trim());
            if (isNaN(seconds)) {
                return [];
            }
            const tick = secondsToTick(seconds);
            if (tick === null || tick < 1) {
                return [];
            }
            spawns.push(tick);
        } else {
            const tick = parseInt(part.trim(), 10);
            if (isNaN(tick) || tick < 1) {
                return [];
            }
            spawns.push(tick);
        }
    }
    spawns.sort((a, b) => a - b);
    return spawns;
}

function useSeconds(): boolean {
    return secondsMode;
}

function tickToDisplay(tick: number): string {
    if (useSeconds()) {
        return ((tick - 1) * 0.6).toFixed(1).replace(/\.0$/, "");
    }
    return String(tick);
}

function secondsToTick(seconds: number): number | null {
    const tick = seconds / 0.6 + 1;
    const rounded = Math.round(tick);
    if (Math.abs(tick - rounded) > 0.001 || rounded < 1) {
        return null;
    }
    return rounded;
}

function convertSpawnsInputToTicks(value: string): Array<number> | null {
    const trimmed = value.trim();
    if (trimmed.length === 0) {
        return [];
    }
    const parts = trimmed.split(/[,\-]/);
    const spawns: Array<number> = [];
    for (const part of parts) {
        const seconds = parseFloat(part.trim());
        if (isNaN(seconds)) {
            return null;
        }
        const tick = secondsToTick(seconds);
        if (tick === null || tick < 1) {
            return null;
        }
        spawns.push(tick);
    }
    spawns.sort((a, b) => a - b);
    return spawns;
}

function getControlledPlayerObject(): Player {
    switch (player) {
        case "defender": return barbarianAssault.defenderPlayer;
        case "mainattacker": return barbarianAssault.mainAttackerPlayer;
        case "secondattacker": return barbarianAssault.secondAttackerPlayer;
        case "healer": return barbarianAssault.healerPlayer;
        case "collector": return barbarianAssault.collectorPlayer;
        default: return null;
    }
}

/**
 * Handles the given keyboard event.
 *
 * @param keyboardEvent the keyboard event to handle
 */
function windowOnKeyDown(keyboardEvent: KeyboardEvent): void {
    const key: string = keyboardEvent.key;

    if (isRunning) {
        if (key !== "p" && key !== "s" && key !== "d" && key !== "f") {
            stateHistory.splice(stateIndex + 1);
        }

        const seedCheckPlayer = getControlledPlayerObject();
        const seedBlocked = seedCheckPlayer !== null && (seedCheckPlayer.seedMovedThisTick || seedCheckPlayer.pendingSeed !== null);

        if (!seedBlocked && simpleFood && player === "defender") {
            switch (key) {
                case "r":
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":r<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "w":
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":w<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "t":
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":t<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "e":
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.shouldPickUpAnyFood = true;
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":e<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
                case "l":
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.isPickingUpLogs = true;
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":l<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    break;
            }
        }

        const code = keyboardEvent.code;
        if (code >= "Digit1" && code <= "Digit8" && !seedBlocked) {
            const controlledPlayer = getControlledPlayerObject();
            if (controlledPlayer !== null) {
                const healerId = Number(code[5]);
                if (keyboardEvent.shiftKey) {
                    controlledPlayer.redXHealerId = healerId;
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":x" + healerId + "<br>";
                } else {
                    controlledPlayer.clearCodeQueue();
                    controlledPlayer.codeQueue.push(new HealerCodeAction(healerId, 0));
                    controlledPlayer.initializeFoodPath(barbarianAssault);
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":h" + healerId + ",1<br>";
                }
                controlledCommands.scrollTop = controlledCommands.scrollHeight;
            }
        }

        switch (key) {
            case "t":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.TOFU);

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":t<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "c":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.CRACKERS);

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":c<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "w":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.dropFood(barbarianAssault, FoodType.WORMS);

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":w<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "u":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":u<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "i":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.CRACKERS;

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":i<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "o":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.foodBeingPickedUp = FoodType.WORMS;

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":o<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "e":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.shouldPickUpAnyFood = true;

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":e<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "l":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.isPickingUpLogs = true;

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":l<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "r":
                if (!seedBlocked && !simpleFood && player === "defender") {
                    barbarianAssault.defenderPlayer.clearCodeQueue();
                    barbarianAssault.defenderPlayer.pathDestination = null;
                    barbarianAssault.defenderPlayer.isRedXPath = false;
                    barbarianAssault.defenderPlayer.startRepairing(barbarianAssault);

                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":r<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            case "m": {
                const controlledPlayer = getControlledPlayerObject();
                if (controlledPlayer !== null) {
                    controlledPlayer.isRunning = !controlledPlayer.isRunning;
                    controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":m<br>";
                    controlledCommands.scrollTop = controlledCommands.scrollHeight;
                }
                break;
            }
            case ",": {
                if (!seedBlocked) {
                    const controlledPlayer = getControlledPlayerObject();
                    if (controlledPlayer !== null) {
                        controlledPlayer.pendingSeed = "MITHRIL";
                        controlledPlayer.clearCodeQueue();
                        if (!toggleSeedQueuePath.checked && lastClickTick !== barbarianAssault.ticks) {
                            controlledPlayer.pathDestination = null;
                            controlledPlayer.isRedXPath = false;
                            controlledPlayer.checkpoints = [];
                            controlledPlayer.checkpointIndex = 0;
                        }
                        controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":,<br>";
                        controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    }
                }
                break;
            }
            case ".": {
                if (!seedBlocked) {
                    const controlledPlayer = getControlledPlayerObject();
                    if (controlledPlayer !== null) {
                        controlledPlayer.pendingSeed = "ADAMANT";
                        controlledPlayer.clearCodeQueue();
                        if (!toggleSeedQueuePath.checked && lastClickTick !== barbarianAssault.ticks) {
                            controlledPlayer.pathDestination = null;
                            controlledPlayer.isRedXPath = false;
                            controlledPlayer.checkpoints = [];
                            controlledPlayer.checkpointIndex = 0;
                        }
                        controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":.<br>";
                        controlledCommands.scrollTop = controlledCommands.scrollHeight;
                    }
                }
                break;
            }
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

    if (key === "y" && saveExists) {
        if (!isRunning) {
            isRunning = true;
            startStopButton.innerHTML = "Stop Wave";
            pauseResumeButton.style.display = "";
            stepBackButton.style.display = "";
            stepForwardButton.style.display = "";
            saveStateButton.disabled = false;
            (document.getElementById(HTML_RUNNER_TABLE) as HTMLElement).style.display = "table";
            (document.getElementById(HTML_HEALER_TABLE) as HTMLElement).style.display = "table";
            tickTimerId = setInterval(tick, Number(tickDurationInput.value));
        }
        isPaused = true;
        pauseResumeButton.innerHTML = "Resume";
        load();
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
    savedFoodCallsString = foodCallsInput.value;
    savedCannonQueueString = cannonQueueInput.value;
    savedHealerSpawnTargetsString = healerSpawnTargetsInput.value;
    savedRunnerSpawnsString = runnerSpawnsInput.value;
    savedHealerSpawnsString = healerSpawnsInput.value;
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
    foodCallsInput.value = savedFoodCallsString;
    cannonQueueInput.value = savedCannonQueueString;
    healerSpawnTargetsInput.value = savedHealerSpawnTargetsString;
    runnerSpawnsInput.value = savedRunnerSpawnsString;
    healerSpawnsInput.value = savedHealerSpawnsString;

    barbarianAssault = savedBarbarianAssault;

    // the existing save state will mutate as the simulator proceeds,
    // so re-clone the save state in case of subsequent loads
    save();

    draw();
}

function pushState(): void {
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

function stepBackward(): void {
    if (stateIndex <= 0) {
        return;
    }
    stateIndex--;
    loadState(stateHistory[stateIndex]);
}

function stepForward(): void {
    if (stateIndex < stateHistory.length - 1) {
        stateIndex++;
        loadState(stateHistory[stateIndex]);
    } else {
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

function loadState(snapshot: {ba: BarbarianAssault, tickHTML: string, foodHTML: string, commandsHTML: string}): void {
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
function canvasOnMouseDown(mouseEvent: MouseEvent): void {
    const canvasRect: DOMRect = renderer.canvas.getBoundingClientRect();
    const xTile: number = Math.trunc((mouseEvent.clientX - canvasRect.left) / renderer.tileSize) + renderer.tileOffsetX;
    const yTile: number = Math.trunc((canvasRect.bottom - 1 - mouseEvent.clientY) / renderer.tileSize);

    if (mouseEvent.button === 0) {
        if (markingTiles) {
            let tileAlreadyMarked: boolean = false;

            for (let i: number = 0; i < markedTiles.length; i++) {
                if ((markedTiles[i].position.x === xTile) && (markedTiles[i].position.y === yTile)) {
                    tileAlreadyMarked = true;
                    markedTiles.splice(i, 1);
                }
            }

            if (!tileAlreadyMarked) {
                markedTiles.push(new TileMarker(new Position(xTile, yTile), RGBColor.fromHexColor(markerColor)));
            }

            saveMarkedTiles();

            if (!isRunning) {
                draw();
            }
        } else {
            const controlledPlayer = getControlledPlayerObject();
            if (controlledPlayer !== null && (controlledPlayer.seedMovedThisTick || controlledPlayer.pendingSeed !== null)) {
                return;
            }

            if (mouseEvent.shiftKey && isRunning && player === "defender") {
                stateHistory.splice(stateIndex + 1);
                barbarianAssault.defenderPlayer.pickUpFoodAtPosition = new Position(xTile, yTile);
                controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":e," + xTile + "," + yTile + "<br>";
                controlledCommands.scrollTop = controlledCommands.scrollHeight;
                return;
            }

            lastClickTick = barbarianAssault.ticks;
            stateHistory.splice(stateIndex + 1);

            const controlledPlayerObj = getControlledPlayerObject();
            if (controlledPlayerObj !== null) {
                controlledPlayerObj.clearCodeQueue();
                controlledPlayerObj.isRedXPath = false;
            }

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

            drawYellowClick(mouseEvent);
            controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":" + xTile + "," + yTile + "<br>";
            controlledCommands.scrollTop = controlledCommands.scrollHeight;
        }
    } else if (mouseEvent.button === 2) {
        if (!isRunning) return;
        const controlledPlayer = getControlledPlayerObject();
        if (controlledPlayer === null) return;
        if (controlledPlayer.seedMovedThisTick || controlledPlayer.pendingSeed !== null) return;

        lastClickTick = barbarianAssault.ticks;
        stateHistory.splice(stateIndex + 1);

        controlledPlayer.clearCodeQueue();
        controlledPlayer.isRedXPath = true;
        controlledPlayer.findPath(barbarianAssault, new Position(xTile, yTile));

        drawYellowClick(mouseEvent);
        controlledCommands.innerHTML += tickToDisplay(barbarianAssault.ticks) + ":>" + xTile + "," + yTile + "<br>";
        controlledCommands.scrollTop = controlledCommands.scrollHeight;
    }
}

function drawYellowClick(e: MouseEvent): void {
    const existing = document.getElementsByClassName("yellow-click");
    if (existing.length) {
        existing[0].remove();
    }
    const el = document.createElement("img");
    el.className = "yellow-click";
    el.src = "static/yellow_click.gif?" + Date.now();
    el.style.left = `${e.clientX - 6}px`;
    el.style.top = `${e.clientY - 6}px`;
    document.body.appendChild(el);
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
    drawEggIcons();
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
        const markedTile: TileMarker = markedTiles[i];

        renderer.setDrawColor(markedTile.rgbColor.r, markedTile.rgbColor.g, markedTile.rgbColor.b, 255);
        renderer.outline(markedTile.position.x, markedTile.position.y);
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
    barbarianAssault.runners.forEach((runner: RunnerPenance): void => {
        if (runner.blueCounter >= 0) {
            renderer.setDrawColor(10, 10, 240, 200);
            renderer.fill(runner.position.x, runner.position.y);
            renderer.setDrawColor(100, 100, 255, 255);
            renderer.outline(runner.position.x, runner.position.y);
        } else if (runner.greenCounter >= 0) {
            renderer.setDrawColor(10, 10, 240, 127);
            renderer.fill(runner.position.x, runner.position.y);
            renderer.setDrawColor(0, 200, 0, 255);
            renderer.outline(runner.position.x, runner.position.y);
        } else if (runner.isDying) {
            renderer.setDrawColor(5, 5, 120, 127);
            renderer.fill(runner.position.x, runner.position.y);
        } else {
            renderer.setDrawColor(10, 10, 240, 127);
            renderer.fill(runner.position.x, runner.position.y);
        }
    });

    barbarianAssault.healers.forEach((healer: HealerPenance): void => {
        if (healer.zombieState) {
            renderer.setDrawColor(80, 80, 80, 180);
            renderer.fill(healer.position.x, healer.position.y);
            renderer.setDrawColor(0, 200, 0, 255);
            renderer.outline(healer.position.x, healer.position.y);
        } else if (healer.blueCounter >= 0) {
            renderer.setDrawColor(10, 240, 10, 127);
            renderer.fill(healer.position.x, healer.position.y);
            renderer.setDrawColor(100, 100, 255, 255);
            renderer.outline(healer.position.x, healer.position.y);
        } else if (healer.isDying) {
            renderer.setDrawColor(5, 120, 5, 127);
            renderer.fill(healer.position.x, healer.position.y);
        } else {
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
function drawGrid(): void {
    for (let xTile: number = 0; xTile < barbarianAssault.map.width; xTile++) {
        if (xTile % 8 === 7) {
            renderer.setDrawColor(40, 22, 12, 175);
        } else {
            renderer.setDrawColor(0, 0, 0, 48);
        }

        renderer.eastLineBig(xTile, 0, barbarianAssault.map.height);
    }

    for (let yTile: number = 0; yTile < barbarianAssault.map.height; yTile++) {
        if (yTile % 8 === 7) {
            renderer.setDrawColor(40, 22, 12, 175);
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

    if (toggleRenderDistance.checked) {
        drawRenderDistance();
    }
}

function drawEggIcons(): void {
    const penance = [...barbarianAssault.runners, ...barbarianAssault.healers];
    const ts = renderer.tileSize;
    const drawn: Record<string, Set<string>> = { w: new Set(), e: new Set() };

    for (const p of penance) {
        for (const egg of p.eggQueue) {
            if (egg.stalled >= 0) {
                if (drawn[egg.cannon].has(egg.type)) continue;
                drawn[egg.cannon].add(egg.type);

                const cannonPos = getCannonPosition(egg.cannon === "w" ? CannonSide.WEST : CannonSide.EAST);
                const img = eggImages[egg.type];
                if (img === undefined || !img.complete) continue;

                const dx = egg.type === EggType.RED ? -10 : egg.type === EggType.GREEN ? -4 : 3;
                const dy = -8;
                const drawX = (cannonPos.x - renderer.tileOffsetX) * ts + dx - 1;
                const drawY = renderer.canvasHeight - (cannonPos.y * ts) - ts + dy;
                const drawW = ts * 1.2;
                const drawH = ts * 1.2;

                renderer.context.drawImage(img, drawX, drawY, drawW, drawH);
            }
        }
    }
}

function drawRenderDistance(): void {
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

        const foodCalls: Array<FoodType> = simpleFood ? [] : parseFoodCallsInput();

        if (foodCalls === null) {
            alert("Invalid food calls. Example: twcw");
            return;
        }

        const mainAttackerCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value, "mainattacker");
        const secondAttackerCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value, "secondattacker");
        const healerCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value, "healer");
        const collectorCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value, "collector");
        const defenderCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value, "defender");

        if (mainAttackerCommands === null || secondAttackerCommands === null || healerCommands === null || collectorCommands === null || defenderCommands === null) {
            alert("Invalid team commands. Example: 7:20,24");
            return;
        }

        const cannonQueue: Array<CannonCommand> = parseCannonInput(cannonQueueInput.value, useSeconds());

        if (cannonQueue === null) {
            alert("Invalid cannon queue. Example: wrr,1,51-erg,1,21");
            return;
        }

        isRunning = true;
        isPaused = false;
        startStopButton.innerHTML = "Stop Wave";
        pauseResumeButton.innerHTML = "Pause";
        pauseResumeButton.style.display = "";
        stepBackButton.style.display = "";
        stepForwardButton.style.display = "";
        saveStateButton.disabled = false;
        (document.getElementById(HTML_RUNNER_TABLE) as HTMLElement).style.display = "table";
        (document.getElementById(HTML_HEALER_TABLE) as HTMLElement).style.display = "table";

        controlledCommands.innerHTML = "";

        barbarianAssault = new BarbarianAssault(
            wave,
            requireRepairs,
            requireLogs,
            simpleFood ? true : infiniteFood,
            movements,
            defenderLevel,
            player === "mainattacker" ? new Map<number, Array<Command>> : mainAttackerCommands,
            player === "secondattacker" ? new Map<number, Array<Command>> : secondAttackerCommands,
            player === "healer" ? new Map<number, Array<Command>> : healerCommands,
            player === "collector" ? new Map<number, Array<Command>> : collectorCommands,
            player === "defender" ? new Map<number, Array<Command>> : defenderCommands,
            foodCalls,
            cannonQueue
        );

        const spawnTargetsValue = healerSpawnTargetsInput.value.trim();
        if (spawnTargetsValue.length > 0) {
            barbarianAssault.healerSpawnTargets = spawnTargetsValue.split("-");
        }

        barbarianAssault.simpleFood = simpleFood;
        barbarianAssault.runnerSpawns = parseSpawnsInput(runnerSpawnsInput.value);
        barbarianAssault.healerSpawns = parseSpawnsInput(healerSpawnsInput.value);
        barbarianAssault.renderDistanceEnabled = toggleRenderDistance.checked;
        barbarianAssault.seedQueuePath = toggleSeedQueuePath.checked;
        barbarianAssault.ignoreMaxHealers = toggleIgnoreMaxHealers.checked;

        // Parse start tick and rapidly simulate to it
        let startTick = 1;
        const startTickValue = startTickInput.value.trim();
        if (startTickValue.length > 0) {
            if (useSeconds()) {
                const sec = parseFloat(startTickValue);
                if (isNaN(sec)) {
                    alert("Invalid start tick value.");
                    reset();
                    return;
                }
                const t = secondsToTick(sec);
                if (t === null) {
                    alert("Invalid start tick: " + startTickValue + " is not a valid time");
                    reset();
                    return;
                }
                startTick = t;
            } else {
                startTick = parseInt(startTickValue);
                if (isNaN(startTick) || startTick < 1) {
                    alert("Invalid start tick value.");
                    reset();
                    return;
                }
            }
        }

        if (startTick > 1) {
            for (let i = 1; i < startTick; i++) {
                barbarianAssault.tick();
                if (!simpleFood) {
                    currentDefenderFoodSpan.innerHTML = barbarianAssault.defenderFoodCall.toString();
                }
                tickCountSpan.innerHTML = barbarianAssault.ticks.toString() + " (" + ticksToSeconds(barbarianAssault.ticks) + "s)";
                pushState();
            }
            draw();
            updateRunnerTable();
            updateHealerTable();
        }

        console.log("Wave " + wave + " started!");
        tick();
        tickTimerId = setInterval(tick, Number(tickDurationInput.value));
    }
}

function simulateButtonOnClick(): void {
    if (isRunning) {
        barbarianAssault.map.reset();
        reset();
    }

    const allRunnerMovementsToCheck: Array<Array<string>> = parseRunnerMovementsToCheck();

    if (allRunnerMovementsToCheck === null) {
        alert("Invalid runner movements to check. Example: ws-x-ex or [sw]e-w-s");
        return;
    }

    const foodCalls: Array<FoodType> = parseFoodCallsInput();

    if (foodCalls === null) {
        alert("Invalid food calls. Example: twcw");
        return;
    }

    let runnersDeadByTick: number;
    if (useSeconds()) {
        const sec = parseFloat(runnersDeadByTickInput.value);
        if (isNaN(sec)) {
            alert("Invalid runners dead by time.");
            return;
        }
        const t = secondsToTick(sec);
        if (t === null) {
            alert("Invalid runners dead by time: " + runnersDeadByTickInput.value + " is not a valid time");
            return;
        }
        runnersDeadByTick = t;
    } else {
        runnersDeadByTick = Number(runnersDeadByTickInput.value);
        if (!Number.isInteger(runnersDeadByTick) || runnersDeadByTick < 1) {
            alert("Invalid runners dead by tick. Example: 12");
            return;
        }
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

    let runnersDoNotDieWithMovementsInnerHTML: string = "";

    for (const runnerMovementsToCheck of allRunnerMovementsToCheck) {
        const movementsRunnersDoNotDieOnTime: Array<Array<string>> = getMovementsRunnersDoNotDieOnTime(foodCalls, runnerMovementsToCheck, runnersDeadByTick);

        for (let i: number = 0; i < movementsRunnersDoNotDieOnTime.length; i++) {
            const movement: Array<string> = movementsRunnersDoNotDieOnTime[i];
            runnersDoNotDieWithMovementsInnerHTML += getMovementsStringFromArray(movement) + "<br>";
        }
    }

    if (runnersDoNotDieWithMovementsInnerHTML === "") {
        runnersDoNotDieWithMovementsInnerHTML = "None!";
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
function updateRunnerTable(): void {
    const table = document.getElementById(HTML_RUNNER_TABLE) as HTMLTableElement;

    const oldBody = table.querySelector("tbody");
    if (oldBody) {
        oldBody.remove();
    }

    const tbody = document.createElement("tbody");

    barbarianAssault.runners.forEach((runner: RunnerPenance): void => {
        const row = tbody.insertRow();
        const cellStyle = "border: 1px solid; padding: 2px 6px;";

        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.id}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.cycleTick}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.targetState}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.hp}/5</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${runner.position.x}, ${runner.position.y})</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${runner.destination.x}, ${runner.destination.y})</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.foodTarget !== null ? "(" + runner.foodTarget.position.x + ", " + runner.foodTarget.position.y + ")" : "None"}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${runner.chat}</td>`;

        let status = "";
        if (runner.isDying) status = "Dying";
        else if (runner.blueCounter >= 0) status = "Stunned";
        else if (runner.greenCounter >= 0) status = "Poisoned";
        else if (runner.blughhhhCountdown > 0) status = "Blughhhh";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${status}</td>`;
    });

    table.appendChild(tbody);
}

function updateHealerTable(): void {
    const table = document.getElementById(HTML_HEALER_TABLE) as HTMLTableElement;

    const oldBody = table.querySelector("tbody");
    if (oldBody) {
        oldBody.remove();
    }

    const tbody = document.createElement("tbody");

    barbarianAssault.healers.forEach((healer: HealerPenance): void => {
        const row = tbody.insertRow();
        const cellStyle = "border: 1px solid; padding: 2px 6px;";

        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.id}</td>`;

        let targetStr = "";
        if (healer.target instanceof Player) {
            if (healer.target === barbarianAssault.mainAttackerPlayer) targetStr = "Main";
            else if (healer.target === barbarianAssault.secondAttackerPlayer) targetStr = "Second";
            else if (healer.target === barbarianAssault.healerPlayer) targetStr = "Healer";
            else if (healer.target === barbarianAssault.collectorPlayer) targetStr = "Collector";
            else if (healer.target === barbarianAssault.defenderPlayer) targetStr = "Defender";
        } else if (healer.target instanceof RunnerPenance) {
            targetStr = "Runner";
        }
        row.insertCell().outerHTML = `<td style="${cellStyle}">${targetStr}</td>`;

        let lastTargetStr = "";
        if (healer.previousTargetType === HealerTargetType.PLAYER) lastTargetStr = "Player";
        else if (healer.previousTargetType === HealerTargetType.RUNNER) lastTargetStr = "Runner";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${lastTargetStr}</td>`;

        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.sprayTimer}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.isPoisoned ? "Yes" : "No"}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">${healer.health}/${healer.maxHealth}</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${healer.position.x}, ${healer.position.y})</td>`;
        row.insertCell().outerHTML = `<td style="${cellStyle}">(${healer.destination.x}, ${healer.destination.y})</td>`;

        let status = "";
        if (healer.isDying) status = "Dying";
        else if (healer.zombieState) status = "Zombie";
        else if (healer.blueCounter >= 0) status = "Stunned";
        else if (healer.greenCounter >= 0) status = "Egg Psn";
        row.insertCell().outerHTML = `<td style="${cellStyle}">${status}</td>`;
    });

    table.appendChild(tbody);
}

function tick(): void {
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
function toggleMarkingTilesOnChange(): void {
    markingTiles = toggleMarkingTiles.checked;
}

function exportMarkers(): void {
    const regionId = wave === 10 ? 7508 : 7509;
    const tiles = markedTiles.map((tile: TileMarker) => ({
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

function importMarkers(): void {
    const field = document.getElementById(HTML_MARKER_IMPORT_FIELD) as HTMLInputElement;
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
        saveMarkedTiles();
        alert("Imported " + count + " tile marker(s).");
        if (!isRunning) {
            draw();
        }
    } catch (e) {
        alert("Failed to parse RuneLite tile marker JSON.");
    }
}

function clearMarkers(): void {
    markedTiles = [];
    saveMarkedTiles();
    if (!isRunning) {
        draw();
    }
}

function exportSettings(): void {
    const settings: Record<string, unknown> = {
        defenderLevel: defenderLevelSelection.value,
        wave: waveSelect.value,
        runnerMovements: movementsInput.value,
        foodCalls: foodCallsInput.value,
        cannonQueue: cannonQueueInput.value,
        healerSpawnTargets: healerSpawnTargetsInput.value,
        runnerSpawns: runnerSpawnsInput.value,
        healerSpawns: healerSpawnsInput.value,
        tickDuration: (document.getElementById(HTML_TICK_DURATION) as HTMLInputElement).value,
        infiniteFood: (document.getElementById(HTML_TOGGLE_INFINITE_FOOD) as HTMLInputElement).checked,
        requireRepairs: (document.getElementById(HTML_TOGGLE_REPAIR) as HTMLInputElement).checked,
        requireLogToRepair: (document.getElementById(HTML_TOGGLE_LOG_TO_REPAIR) as HTMLInputElement).checked,
        renderDistance: (document.getElementById(HTML_TOGGLE_RENDER_DISTANCE) as HTMLInputElement).checked,
        seedQueuePath: (document.getElementById(HTML_TOGGLE_SEED_QUEUE_PATH) as HTMLInputElement).checked,
        ignoreMaxHealers: (document.getElementById(HTML_TOGGLE_IGNORE_MAX_HEALERS) as HTMLInputElement).checked,
        simpleFood: simpleFood,
        mainAttacker: (document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLTextAreaElement).value,
        secondAttacker: (document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLTextAreaElement).value,
        healer: (document.getElementById(HTML_HEALER_COMMANDS) as HTMLTextAreaElement).value,
        collector: (document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLTextAreaElement).value,
        defender: (document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLTextAreaElement).value,
        playerToControl: playerSelect.value,
        secondsMode: secondsMode,
        startTick: startTickInput.value,
        runnerMovementsToCheck: runnerMovementsToCheckInput.value,
        runnersDeadByTick: runnersDeadByTickInput.value,
    };
    const json = JSON.stringify(settings);
    navigator.clipboard.writeText(json);
    alert("Settings copied to clipboard.");
}

function importSettings(): void {
    const field = document.getElementById(HTML_SETTINGS_IMPORT_FIELD) as HTMLInputElement;
    const input = field.value.trim();
    if (input.length === 0) {
        return;
    }
    try {
        const s = JSON.parse(input);
        if (s.defenderLevel !== undefined) defenderLevelSelection.value = s.defenderLevel;
        if (s.wave !== undefined) {
            waveSelect.value = s.wave;
            wave = Number(waveSelect.value);
        }
        if (s.runnerMovements !== undefined) movementsInput.value = s.runnerMovements;
        if (s.foodCalls !== undefined) foodCallsInput.value = s.foodCalls;
        if (s.cannonQueue !== undefined) cannonQueueInput.value = s.cannonQueue;
        if (s.healerSpawnTargets !== undefined) healerSpawnTargetsInput.value = s.healerSpawnTargets;
        if (s.runnerSpawns !== undefined) runnerSpawnsInput.value = s.runnerSpawns;
        if (s.healerSpawns !== undefined) healerSpawnsInput.value = s.healerSpawns;
        if (s.tickDuration !== undefined) (document.getElementById(HTML_TICK_DURATION) as HTMLInputElement).value = s.tickDuration;
        if (s.infiniteFood !== undefined) (document.getElementById(HTML_TOGGLE_INFINITE_FOOD) as HTMLInputElement).checked = s.infiniteFood;
        if (s.requireRepairs !== undefined) (document.getElementById(HTML_TOGGLE_REPAIR) as HTMLInputElement).checked = s.requireRepairs;
        if (s.requireLogToRepair !== undefined) (document.getElementById(HTML_TOGGLE_LOG_TO_REPAIR) as HTMLInputElement).checked = s.requireLogToRepair;
        if (s.renderDistance !== undefined) (document.getElementById(HTML_TOGGLE_RENDER_DISTANCE) as HTMLInputElement).checked = s.renderDistance;
        if (s.seedQueuePath !== undefined) (document.getElementById(HTML_TOGGLE_SEED_QUEUE_PATH) as HTMLInputElement).checked = s.seedQueuePath;
        if (s.ignoreMaxHealers !== undefined) (document.getElementById(HTML_TOGGLE_IGNORE_MAX_HEALERS) as HTMLInputElement).checked = s.ignoreMaxHealers;
        if (s.simpleFood !== undefined && s.simpleFood !== simpleFood) {
            (document.getElementById(HTML_TOGGLE_SIMPLE_FOOD) as HTMLButtonElement).click();
        }
        if (s.mainAttacker !== undefined) (document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLTextAreaElement).value = s.mainAttacker;
        if (s.secondAttacker !== undefined) (document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLTextAreaElement).value = s.secondAttacker;
        if (s.healer !== undefined) (document.getElementById(HTML_HEALER_COMMANDS) as HTMLTextAreaElement).value = s.healer;
        if (s.collector !== undefined) (document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLTextAreaElement).value = s.collector;
        if (s.defender !== undefined) (document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLTextAreaElement).value = s.defender;
        if (s.playerToControl !== undefined) {
            playerSelect.value = s.playerToControl;
            player = playerSelect.value;
        }
        if (s.startTick !== undefined) startTickInput.value = s.startTick;
        if (s.runnerMovementsToCheck !== undefined) runnerMovementsToCheckInput.value = s.runnerMovementsToCheck;
        if (s.runnersDeadByTick !== undefined) runnersDeadByTickInput.value = s.runnersDeadByTick;
        if (s.secondsMode !== undefined) {
            secondsMode = !!s.secondsMode;
            toggleSecondsButton.innerHTML = secondsMode ? "Express time in ticks" : "Express time in seconds";
            startTickInput.placeholder = secondsMode ? "0" : "1";
            document.getElementById("startticklabel").innerHTML = secondsMode ? "Start time" : "Start tick";
            runnerSpawnsInput.placeholder = secondsMode ? "6,12,18" : "11,21,31";
            healerSpawnsInput.placeholder = secondsMode ? "6,12,18" : "11,21,31";
            document.getElementById("runnersdeadbylabel").innerHTML = secondsMode ? "Runners dead by time" : "Runners dead by tick";
            runnersDeadByTickInput.placeholder = secondsMode ? "24.6" : "42";
            const prefixSpans = document.getElementsByClassName("tc-prefix");
            for (let i = 0; i < prefixSpans.length; i++) {
                prefixSpans[i].textContent = secondsMode ? "time" : "tick";
            }
        }
        field.value = "";
        alert("Settings imported.");
    } catch (e) {
        alert("Failed to parse settings JSON.");
    }
}

import {
    solarCommandsToMclovin,
    solarDefenderCommandsToMclovin,
    solarHealerSpawnsToMclovin,
} from "./SolarInterop.js";

function importSettingsFromSolar(): void {
    const field = document.getElementById("solarimportfield") as HTMLInputElement;
    const input = field.value.trim();
    if (input.length === 0) {
        return;
    }
    try {
        const s = JSON.parse(input);
        if (secondsMode) {
            toggleSecondsOnClick();
        }
        if (!simpleFood) {
            (document.getElementById(HTML_TOGGLE_SIMPLE_FOOD) as HTMLButtonElement).click();
        }
        if (s.wave !== undefined) {
            waveSelect.value = s.wave;
            wave = Number(waveSelect.value);
        }
        if (s.runnerMovements !== undefined) movementsInput.value = s.runnerMovements;
        if (s.eggs !== undefined) cannonQueueInput.value = s.eggs;
        if (s.runnerSpawns !== undefined) runnerSpawnsInput.value = s.runnerSpawns;
        if (s.healerSpawns !== undefined) {
            const parsed = solarHealerSpawnsToMclovin(s.healerSpawns);
            healerSpawnsInput.value = parsed.spawns;
            healerSpawnTargetsInput.value = parsed.targets;
        }
        if (s.toggleRender !== undefined) {
            toggleRenderDistance.checked = s.toggleRender;
        }
        if (s.team !== undefined) {
            if (s.team.main !== undefined) (document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLTextAreaElement).value = solarCommandsToMclovin(s.team.main);
            if (s.team.second !== undefined) (document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLTextAreaElement).value = solarCommandsToMclovin(s.team.second);
            if (s.team.heal !== undefined) (document.getElementById(HTML_HEALER_COMMANDS) as HTMLTextAreaElement).value = solarCommandsToMclovin(s.team.heal);
            if (s.team.col !== undefined) (document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLTextAreaElement).value = solarCommandsToMclovin(s.team.col);
            if (s.team.def !== undefined) (document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLTextAreaElement).value = solarDefenderCommandsToMclovin(s.team.def);
        }
        field.value = "";
        alert("Settings imported from Solar sim format.");
    } catch (e) {
        alert("Failed to parse Solar sim settings JSON.");
    }
}

function rgbToHex(color: RGBColor): string {
    return ((1 << 24) | (color.r << 16) | (color.g << 8) | color.b).toString(16).slice(1);
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

function convertDefenderCommands(toSimple: boolean): void {
    const normalToSimple: Record<string, string> = {"t": "r", "c": "w", "r": "t"};
    const simpleToNormal: Record<string, string> = {"r": "t", "w": "c", "t": "r"};
    const map = toSimple ? normalToSimple : simpleToNormal;

    function convertCommandText(text: string): string {
        const lines = text.split("\n");
        const converted: Array<string> = [];
        for (const line of lines) {
            if (line.trim().length === 0) {
                converted.push(line);
                continue;
            }
            const colonIdx = line.indexOf(":");
            if (colonIdx === -1) {
                converted.push(line);
                continue;
            }
            const cmd = line.substring(colonIdx + 1);
            if (cmd in map) {
                converted.push(line.substring(0, colonIdx + 1) + map[cmd]);
            } else {
                converted.push(line);
            }
        }
        return converted.join("\n");
    }

    // Convert defender team commands
    const defenderTextarea = document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLTextAreaElement;
    defenderTextarea.value = convertCommandText(defenderTextarea.value);

    // Convert controlled commands (if defender is controlled)
    const html = controlledCommands.innerHTML;
    if (html.trim().length > 0) {
        const entries = html.split("<br>");
        const converted: Array<string> = [];
        for (const entry of entries) {
            const trimmed = entry.trim();
            if (trimmed.length === 0) continue;
            const colonIdx = trimmed.indexOf(":");
            if (colonIdx === -1) {
                converted.push(trimmed);
                continue;
            }
            const cmd = trimmed.substring(colonIdx + 1);
            if (cmd in map) {
                converted.push(trimmed.substring(0, colonIdx + 1) + map[cmd]);
            } else {
                converted.push(trimmed);
            }
        }
        controlledCommands.innerHTML = converted.map(e => e + "<br>").join("");
    }
}

function toggleSecondsOnClick(): void {
    const toSeconds = !secondsMode;

    // Convert start tick input
    const startTickTrimmed = startTickInput.value.trim();
    if (startTickTrimmed.length > 0) {
        const num = parseFloat(startTickTrimmed);
        if (!isNaN(num)) {
            if (toSeconds) {
                startTickInput.value = ((num - 1) * 0.6).toFixed(1).replace(/\.0$/, "");
            } else {
                const tick = secondsToTick(num);
                if (tick === null) {
                    alert("Cannot convert to ticks: " + num + " is not a valid time");
                    return;
                }
                startTickInput.value = String(tick);
            }
        }
    }

    // Convert runners dead by tick input
    const deadByTrimmed = runnersDeadByTickInput.value.trim();
    if (deadByTrimmed.length > 0) {
        const num = parseFloat(deadByTrimmed);
        if (!isNaN(num)) {
            if (toSeconds) {
                runnersDeadByTickInput.value = ((num - 1) * 0.6).toFixed(1).replace(/\.0$/, "");
            } else {
                const tick = secondsToTick(num);
                if (tick === null) {
                    alert("Cannot convert to ticks: " + num + " is not a valid time");
                    return;
                }
                runnersDeadByTickInput.value = String(tick);
            }
        }
    }

    // Convert spawns inputs
    for (const input of [runnerSpawnsInput, healerSpawnsInput]) {
        const trimmed = input.value.trim();
        if (trimmed.length === 0) continue;
        const separator = trimmed.includes("-") ? "-" : ",";
        const parts = trimmed.split(/[,\-]/);
        const converted: Array<string> = [];
        for (const part of parts) {
            const num = parseFloat(part.trim());
            if (isNaN(num)) return;
            if (toSeconds) {
                converted.push(((num - 1) * 0.6).toFixed(1).replace(/\.0$/, ""));
            } else {
                const tick = secondsToTick(num);
                if (tick === null) {
                    alert("Cannot convert to ticks: " + num + " is not a valid time");
                    return;
                }
                converted.push(String(tick));
            }
        }
        input.value = converted.join(separator);
    }

    // Convert cannon queue
    const cannonTrimmed = cannonQueueInput.value.trim();
    if (cannonTrimmed.length > 0) {
        const cannonParts = cannonTrimmed.split("-");
        const convertedParts: Array<string> = [];
        for (const part of cannonParts) {
            const tokens = part.trim().split(",");
            const lastIdx = tokens.length - 1;
            const num = parseFloat(tokens[lastIdx].trim());
            if (isNaN(num)) return;
            if (toSeconds) {
                tokens[lastIdx] = ((num - 1) * 0.6).toFixed(1).replace(/\.0$/, "");
            } else {
                const tick = secondsToTick(num);
                if (tick === null) {
                    alert("Cannot convert to ticks: " + num + " is not a valid time");
                    return;
                }
                tokens[lastIdx] = String(tick);
            }
            convertedParts.push(tokens.join(","));
        }
        cannonQueueInput.value = convertedParts.join("-");
    }

    // Convert team commands
    const commandInputIds = [
        HTML_MAIN_ATTACKER_COMMANDS,
        HTML_SECOND_ATTACKER_COMMANDS,
        HTML_HEALER_COMMANDS,
        HTML_COLLECTOR_COMMANDS,
        HTML_DEFENDER_COMMANDS,
    ];
    for (const id of commandInputIds) {
        const textarea = document.getElementById(id) as HTMLTextAreaElement;
        const lines = textarea.value.split("\n");
        const convertedLines: Array<string> = [];
        for (const line of lines) {
            if (line.trim().length === 0) {
                convertedLines.push(line);
                continue;
            }
            const colonIdx = line.indexOf(":");
            if (colonIdx === -1) {
                convertedLines.push(line);
                continue;
            }
            const num = parseFloat(line.substring(0, colonIdx));
            if (isNaN(num)) {
                convertedLines.push(line);
                continue;
            }
            const rest = line.substring(colonIdx);
            if (toSeconds) {
                convertedLines.push(((num - 1) * 0.6).toFixed(1).replace(/\.0$/, "") + rest);
            } else {
                const tick = secondsToTick(num);
                if (tick === null) {
                    alert("Cannot convert to ticks: " + num + " is not a valid time");
                    return;
                }
                convertedLines.push(tick + rest);
            }
        }
        textarea.value = convertedLines.join("\n");
    }

    // Convert controlled commands output
    const commandsDiv = controlledCommands;
    const html = commandsDiv.innerHTML;
    if (html.trim().length > 0) {
        const entries = html.split("<br>");
        const convertedEntries: Array<string> = [];
        for (const entry of entries) {
            const trimmedEntry = entry.trim();
            if (trimmedEntry.length === 0) continue;
            const colonIdx = trimmedEntry.indexOf(":");
            if (colonIdx === -1) {
                convertedEntries.push(trimmedEntry);
                continue;
            }
            const num = parseFloat(trimmedEntry.substring(0, colonIdx));
            if (isNaN(num)) {
                convertedEntries.push(trimmedEntry);
                continue;
            }
            const rest = trimmedEntry.substring(colonIdx);
            if (toSeconds) {
                convertedEntries.push(((num - 1) * 0.6).toFixed(1).replace(/\.0$/, "") + rest);
            } else {
                const tick = secondsToTick(num);
                if (tick === null) {
                    convertedEntries.push(trimmedEntry);
                    continue;
                }
                convertedEntries.push(tick + rest);
            }
        }
        commandsDiv.innerHTML = convertedEntries.map(e => e + "<br>").join("");
    }

    secondsMode = toSeconds;
    toggleSecondsButton.innerHTML = secondsMode ? "Express time in ticks" : "Express time in seconds";
    startTickInput.placeholder = secondsMode ? "0" : "1";
    document.getElementById("startticklabel").innerHTML = secondsMode ? "Start time" : "Start tick";
    runnerSpawnsInput.placeholder = secondsMode ? "6,12,18" : "11,21,31";
    healerSpawnsInput.placeholder = secondsMode ? "6,12,18" : "11,21,31";
    document.getElementById("runnersdeadbylabel").innerHTML = secondsMode ? "Runners dead by time" : "Runners dead by tick";
    runnersDeadByTickInput.placeholder = secondsMode ? "24.6" : "42";
    const prefixSpans = document.getElementsByClassName("tc-prefix");
    for (let i = 0; i < prefixSpans.length; i++) {
        prefixSpans[i].textContent = secondsMode ? "time" : "tick";
    }
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

function runnerMovementsToCheckInputOnChange(): void {
    reset();
}

function runnersDeadByTickInputOnChange(): void {
    reset();
}

function foodCallsInputOnChange(): void {
    reset();
}

function markerColorInputOnChange(): void {
    markerColor = Number("0x" + markerColorInput.value.substring(1));
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
 * If the given commands string is invalid, then null is returned
 *
 * @param commandsString    the commands string to convert to a map from tick
 *                          numbers to positions
 * @return                  a map from tick numbers to positions as specified by
 *                          the given commands string, or null if the given
 *                          commands string is invalid
 */
function convertCommandsStringToMap(commandsString: string, player: string): Map<number, Array<Command>> {
    if (commandsString === null) {
        return null;
    }

    const commandsMap: Map<number, Array<Command>> = new Map<number, Array<Command>>();

    const commands: Array<string> = commandsString.split("\n");

    let previousCommandTick: number = -1;

    for (let i: number = 0; i < commands.length; i++) {
        const command: string = commands[i];

        if (command.length === 0) {
            continue;
        }

        const tickAndCommand: Array<string> = command.split(":");

        if (tickAndCommand.length !== 2) {
            return null;
        }

        let tick: number;
        if (useSeconds()) {
            const seconds = parseFloat(tickAndCommand[0]);
            if (isNaN(seconds)) {
                return null;
            }
            tick = secondsToTick(seconds);
            if (tick === null || tick < 1 || tick < previousCommandTick) {
                return null;
            }
        } else {
            tick = Number(tickAndCommand[0]);
            if (!Number.isInteger(tick) || tick < 1 || tick < previousCommandTick) {
                return null;
            }
        }

        if (tickAndCommand[1] === ",") {
            addToCommandsMap(commandsMap, tick, new SeedCommand("MITHRIL"));
            previousCommandTick = tick;
            continue;
        } else if (tickAndCommand[1] === ".") {
            addToCommandsMap(commandsMap, tick, new SeedCommand("ADAMANT"));
            previousCommandTick = tick;
            continue;
        }

        const commandTokens: Array<string> = tickAndCommand[1].split(",");

        if (commandTokens.length === 1) {
            if (commandTokens[0] === "walk") {
                addToCommandsMap(commandsMap, tick, new WalkRunCommand(false));
            } else if (commandTokens[0] === "run") {
                addToCommandsMap(commandsMap, tick, new WalkRunCommand(true));
            } else if (commandTokens[0] === "m") {
                addToCommandsMap(commandsMap, tick, new ToggleRunCommand());
            } else if (/^x[1-8]$/.test(commandTokens[0])) {
                addToCommandsMap(commandsMap, tick, new RedXCommand(Number(commandTokens[0][1])));
            } else if (player !== "defender") {
                return null;
            } else if (simpleFood) {
                switch (commandTokens[0]) {
                    case "r":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.DROP_TOFU));
                        break;
                    case "w":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.DROP_CRACKERS));
                        break;
                    case "t":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.REPAIR_TRAP));
                        break;
                    case "e":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_ANY_FOOD));
                        break;
                    case "l":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_LOGS));
                        break;
                    default:
                        return null;
                }
            } else {
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
                    case "u":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_TOFU));
                        break;
                    case "i":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_CRACKERS));
                        break;
                    case "o":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_WORMS));
                        break;
                    case "e":
                        addToCommandsMap(commandsMap, tick, new DefenderActionCommand(DefenderActionType.PICKUP_ANY_FOOD));
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
        } else if (commandTokens.length === 2) {
            const healerMatch = commandTokens[0].match(/^h(\d+)$/);
            if (healerMatch) {
                const healerId = parseInt(healerMatch[1]);
                const count = parseInt(commandTokens[1]);
                if (!Number.isInteger(healerId) || !Number.isInteger(count) || healerId < 1 || count < 1) {
                    return null;
                }
                addToCommandsMap(commandsMap, tick, new HealerCodeCommand(healerId, count));
            } else if (commandTokens[0].startsWith(">")) {
                const positionX: number = Number(commandTokens[0].substring(1));
                const positionY: number = Number(commandTokens[1]);

                if (!Number.isInteger(positionX) || !Number.isInteger(positionY)) {
                    return null;
                }

                addToCommandsMap(commandsMap, tick, new RedXMoveCommand(new Position(positionX, positionY)));
            } else {
                const positionX: number = Number(commandTokens[0]);
                const positionY: number = Number(commandTokens[1]);

                if (!Number.isInteger(positionX) || !Number.isInteger(positionY)) {
                    return null;
                }

                addToCommandsMap(commandsMap, tick, new MoveCommand(new Position(positionX, positionY)));
            }
        } else if (commandTokens.length === 3) {
            if (commandTokens[0] === "e" && player === "defender") {
                const positionX: number = Number(commandTokens[1]);
                const positionY: number = Number(commandTokens[2]);

                if (!Number.isInteger(positionX) || !Number.isInteger(positionY)) {
                    return null;
                }

                addToCommandsMap(commandsMap, tick, new DefenderPickupAtCommand(new Position(positionX, positionY)));
            } else {
                return null;
            }
        } else {
            return null;
        }

        previousCommandTick = tick;
    }

    return commandsMap;
}

function addToCommandsMap(commandsMap: Map<number, Array<Command>>, tick: number, command: Command): void {
    if (commandsMap.has(tick)) {
        commandsMap.get(tick).push(command);
    } else {
        commandsMap.set(tick, [command]);
    }
}

/**
 * Converts time measured in ticks to seconds.
 *
 * @param ticks the number of ticks to convert to seconds
 */
function ticksToSeconds(ticks: number): string {
    return (0.6 * Math.max(ticks - 1, 0)).toFixed(1);
}

function runnersDieOnTimeForMovements(
    runnerMovements: Array<string>,
    foodCalls: Array<FoodType>,
    runnersDeadByTick: number,
    mainAttackerCommands: Map<number, Array<Command>>,
    secondAttackerCommands: Map<number, Array<Command>>,
    healerCommands: Map<number, Array<Command>>,
    collectorCommands: Map<number, Array<Command>>,
    defenderCommands: Map<number, Array<Command>>
): boolean {
    const cannonQueue: Array<CannonCommand> = parseCannonInput(cannonQueueInput.value, useSeconds());

    const barbarianAssaultSim: BarbarianAssault = new BarbarianAssault(
        wave,
        requireRepairs,
        requireLogs,
        infiniteFood,
        runnerMovements,
        defenderLevel,
        mainAttackerCommands,
        secondAttackerCommands,
        healerCommands,
        collectorCommands,
        defenderCommands,
        foodCalls,
        cannonQueue || []
    );
    barbarianAssaultSim.simpleFood = simpleFood;
    barbarianAssaultSim.ignoreMaxHealers = toggleIgnoreMaxHealers.checked;
    barbarianAssaultSim.runnerSpawns = parseSpawnsInput(runnerSpawnsInput.value);
    barbarianAssaultSim.healerSpawns = parseSpawnsInput(healerSpawnsInput.value);

    for (let i: number = 0; i < runnersDeadByTick; i++) {
        barbarianAssaultSim.tick();
    }

    return barbarianAssaultSim.runnersKilled >= barbarianAssaultSim.totalRunners;
}

function getMovementsRunnersDoNotDieOnTime(foodCalls: Array<FoodType>, runnerMovementsToCheck: Array<string>, runnersDeadByTick: number): Array<Array<string>> {
    const movementsRunnersDoNotDieOnTime: Array<Array<string>> = [];

    const candidateMovements: Array<Array<string>> = [];

    runnerMovementsToCheck.forEach((movementPattern: string): void => {
        candidateMovements.push(getAllForcedMovementsForOneRunner(movementPattern));
    });

    const allCombinations: Array<Array<string>> = getAllCombinations(candidateMovements, 0, [[]]);

    const mainAttackerCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_MAIN_ATTACKER_COMMANDS) as HTMLInputElement).value, "mainattacker");
    const secondAttackerCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_SECOND_ATTACKER_COMMANDS) as HTMLInputElement).value, "secondattacker");
    const healerCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_HEALER_COMMANDS) as HTMLInputElement).value, "healer");
    const collectorCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_COLLECTOR_COMMANDS) as HTMLInputElement).value, "collector");
    const defenderCommands: Map<number, Array<Command>> = convertCommandsStringToMap((document.getElementById(HTML_DEFENDER_COMMANDS) as HTMLInputElement).value, "defender");

    for (let i: number = 0; i < allCombinations.length; i++) {
        const runnerMovements: Array<string> = allCombinations[i];

        if (!runnersDieOnTimeForMovements(runnerMovements, foodCalls, runnersDeadByTick, mainAttackerCommands, secondAttackerCommands, healerCommands, collectorCommands, defenderCommands)) {
            movementsRunnersDoNotDieOnTime.push(runnerMovements);
        }
    }

    return movementsRunnersDoNotDieOnTime;
}

function getAllCombinations(candidateMovements: Array<Array<string>>, index: number, partialCombinations: Array<Array<string>>): Array<Array<string>> {
   if (index >= candidateMovements.length) {
       return partialCombinations;
   }

   const newPartialCombinations: Array<Array<string>> = [];

   partialCombinations.forEach((partialCombination: Array<string>): void => {
       candidateMovements[index].forEach((candidateMovement: string): void => {
           newPartialCombinations.push([...partialCombination, candidateMovement]);
       });
   });

   return getAllCombinations(candidateMovements, index + 1, newPartialCombinations);
}

function getAllForcedMovementsForOneRunner(movementPattern: string): Array<string> {
    const validDirections: Array<Array<string>> = [];

    let i: number = 0;
    while (i < movementPattern.length) {
        const c: string = movementPattern[i];
        if (c === "[") {
            i++;
            const dirs: Array<string> = [];
            while (i < movementPattern.length && movementPattern[i] !== "]") {
                dirs.push(movementPattern[i]);
                i++;
            }
            i++;
            validDirections.push(dirs);
        } else if (c === "x") {
            validDirections.push(["s", "e", "w"]);
            i++;
        } else {
            validDirections.push([c]);
            i++;
        }
    }

    return getAllValidPermutationsForOneRunner(validDirections, 0, [""]);
}

function getAllValidPermutationsForOneRunner(validDirections: Array<Array<string>>, index: number, partialMovements: Array<string>): Array<string> {
    if (index >= validDirections.length) {
        return partialMovements;
    }

    const newPartialMovements: Array<string> = [];

    partialMovements.forEach((partialMovement: string): void => {
        validDirections[index].forEach((validDirection: string): void => {
            newPartialMovements.push(partialMovement + validDirection);
        });
    });

    return getAllValidPermutationsForOneRunner(validDirections, index + 1, newPartialMovements);
}

function getMovementsStringFromArray(movementsArray: Array<string>): string {
    let movementsString: string = "";

    for (let i: number = 0; i < movementsArray.length; i++) {
        if (i > 0) {
            movementsString += "-";
        }

        movementsString += movementsArray[i];
    }

    return movementsString;
}