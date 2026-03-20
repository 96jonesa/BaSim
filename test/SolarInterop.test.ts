import {describe, expect, it} from "vitest";
import {
    solarCommandsToMclovin,
    solarDefenderCommandsToMclovin,
    solarHealerSpawnsToMclovin,
} from "../src/SolarInterop";

describe("solarCommandsToMclovin", () => {
    it("converts x,y:tick to tick:x,y", () => {
        expect(solarCommandsToMclovin("20,21:5")).toBe("5:20,21");
    });

    it("adds tick 0 for lines without tick", () => {
        expect(solarCommandsToMclovin("20,21")).toBe("0:20,21");
    });

    it("handles multiple lines", () => {
        expect(solarCommandsToMclovin("20,21:5\n30,31:10")).toBe("5:20,21\n10:30,31");
    });

    it("handles healer code format", () => {
        expect(solarCommandsToMclovin("h1,3:11")).toBe("11:h1,3");
    });
});

describe("solarDefenderCommandsToMclovin", () => {
    it("converts g,1 to r keeping solar tick", () => {
        expect(solarDefenderCommandsToMclovin("g,1:4")).toBe("4:r");
    });

    it("converts b,1 to w keeping solar tick", () => {
        expect(solarDefenderCommandsToMclovin("b,1:4")).toBe("4:w");
    });

    it("converts t to t keeping solar tick", () => {
        expect(solarDefenderCommandsToMclovin("t:4")).toBe("4:t");
    });

    it("converts p to e keeping solar tick", () => {
        expect(solarDefenderCommandsToMclovin("p:4")).toBe("4:e");
    });

    it("expands g,count into multiple r lines", () => {
        expect(solarDefenderCommandsToMclovin("g,3:4")).toBe("4:r\n4:r\n4:r");
    });

    it("expands b,count into multiple w lines", () => {
        expect(solarDefenderCommandsToMclovin("b,2:4")).toBe("4:w\n4:w");
    });

    it("discards no-tick action (solar tick 0)", () => {
        expect(solarDefenderCommandsToMclovin("g,1")).toBe("");
    });

    it("discards no-tick t (solar tick 0)", () => {
        expect(solarDefenderCommandsToMclovin("t")).toBe("");
    });

    it("discards solar tick 1 action", () => {
        expect(solarDefenderCommandsToMclovin("g,1:1")).toBe("");
    });

    it("move commands have tick -1", () => {
        expect(solarDefenderCommandsToMclovin("25,17:5")).toBe("4:25,17");
    });

    it("discards move with solar tick 1", () => {
        expect(solarDefenderCommandsToMclovin("25,17:1")).toBe("");
    });

    it("move without tick is discarded", () => {
        expect(solarDefenderCommandsToMclovin("25,17")).toBe("");
    });

    it("handles multiple mixed commands", () => {
        const input = "25,17:3\ng,1:4\nb,1:9\nt:14\np:19";
        const expected = "2:25,17\n4:r\n9:w\n14:t\n19:e";
        expect(solarDefenderCommandsToMclovin(input)).toBe(expected);
    });

    it("handles large tick values", () => {
        expect(solarDefenderCommandsToMclovin("g,1:99")).toBe("99:r");
    });

    it("solar tick 2 action becomes mclovin tick 2", () => {
        expect(solarDefenderCommandsToMclovin("g,1:2")).toBe("2:r");
    });
});

describe("solarHealerSpawnsToMclovin", () => {
    it("splits combined format without targets", () => {
        expect(solarHealerSpawnsToMclovin("11,21,31")).toEqual({spawns: "11,21,31", targets: ""});
    });

    it("splits combined format with targets", () => {
        expect(solarHealerSpawnsToMclovin("11:mc,21:h,31")).toEqual({spawns: "11,21,31", targets: "mc-h"});
    });

    it("splits combined format with all targets", () => {
        expect(solarHealerSpawnsToMclovin("11:m,21:c,31:h")).toEqual({spawns: "11,21,31", targets: "m-c-h"});
    });

    it("handles single spawn with target", () => {
        expect(solarHealerSpawnsToMclovin("11:m")).toEqual({spawns: "11", targets: "m"});
    });

    it("handles single spawn without target", () => {
        expect(solarHealerSpawnsToMclovin("11")).toEqual({spawns: "11", targets: ""});
    });
});

describe("integration: solar → mclovin parsing", () => {
    it("converted actions parse correctly in mclovin", () => {
        const mc = solarDefenderCommandsToMclovin("g,1:4\nb,1:9\nt:14\np:19");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(4)).toEqual(["r"]);
        expect(parsed!.get(9)).toEqual(["w"]);
        expect(parsed!.get(14)).toEqual(["t"]);
        expect(parsed!.get(19)).toEqual(["e"]);
    });

    it("converted moves parse correctly in mclovin", () => {
        const mc = solarDefenderCommandsToMclovin("25,17:3\n30,22:8");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(2)).toEqual(["25,17"]);
        expect(parsed!.get(7)).toEqual(["30,22"]);
    });

    it("expanded g,count produces valid mclovin output", () => {
        const mc = solarDefenderCommandsToMclovin("g,3:4");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(4)).toEqual(["r", "r", "r"]);
    });

    it("out-of-order ticks cause mclovin parse failure", () => {
        const mc = solarDefenderCommandsToMclovin("g,1:10\n25,17:3");
        expect(mc).toBe("10:r\n2:25,17");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).toBeNull();
    });
});

/**
 * Simulate mclovin's convertCommandsStringToMap() for defender commands (simplified).
 * Returns null if parsing fails, otherwise a map of tick → command strings.
 */
function mclovinParseDefenderCommands(commandsString: string): Map<number, string[]> | null {
    const commandsMap = new Map<number, string[]>();
    const commands = commandsString.split("\n");
    let previousCommandTick = -1;
    for (const command of commands) {
        if (command.length === 0) continue;
        const tickAndCommand = command.split(":");
        if (tickAndCommand.length !== 2) return null;
        const tick = Number(tickAndCommand[0]);
        if (!Number.isInteger(tick) || tick < 1 || tick < previousCommandTick) return null;
        if (!commandsMap.has(tick)) commandsMap.set(tick, []);
        commandsMap.get(tick)!.push(tickAndCommand[1]);
        previousCommandTick = tick;
    }
    return commandsMap;
}
