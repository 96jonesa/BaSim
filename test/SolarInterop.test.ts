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
    it("converts g,1 to r with tick offset", () => {
        expect(solarDefenderCommandsToMclovin("g,1:4")).toBe("5:r");
    });

    it("converts b,1 to w with tick offset", () => {
        expect(solarDefenderCommandsToMclovin("b,1:4")).toBe("5:w");
    });

    it("converts t to t with tick offset", () => {
        expect(solarDefenderCommandsToMclovin("t:4")).toBe("5:t");
    });

    it("converts p to e with tick offset", () => {
        expect(solarDefenderCommandsToMclovin("p:4")).toBe("5:e");
    });

    it("expands g,count into multiple r lines", () => {
        expect(solarDefenderCommandsToMclovin("g,3:4")).toBe("5:r\n5:r\n5:r");
    });

    it("expands b,count into multiple w lines", () => {
        expect(solarDefenderCommandsToMclovin("b,2:4")).toBe("5:w\n5:w");
    });

    it("no-tick action defaults to mclovin tick 1", () => {
        expect(solarDefenderCommandsToMclovin("g,1")).toBe("1:r");
    });

    it("no-tick t defaults to mclovin tick 1", () => {
        expect(solarDefenderCommandsToMclovin("t")).toBe("1:t");
    });

    it("move commands have no tick offset", () => {
        expect(solarDefenderCommandsToMclovin("25,17:5")).toBe("5:25,17");
    });

    it("move without tick becomes tick 0", () => {
        expect(solarDefenderCommandsToMclovin("25,17")).toBe("0:25,17");
    });

    it("handles multiple mixed commands", () => {
        const input = "25,17:3\ng,1:4\nb,1:9\nt:14\np:19";
        const expected = "3:25,17\n5:r\n10:w\n15:t\n20:e";
        expect(solarDefenderCommandsToMclovin(input)).toBe(expected);
    });

    it("handles large tick values", () => {
        expect(solarDefenderCommandsToMclovin("g,1:99")).toBe("100:r");
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

/**
 * Solar-side copy of solarDefCmdsToMclovin (from solar-ba-sim/js/simstate.js)
 * to verify both implementations produce identical output.
 */
function solar_solarDefCmdsToMclovin(commands: string): string {
    const result: string[] = [];
    for (let line of commands.split("\n")) {
        line = line.trim();
        if (!line) continue;
        const tokens = line.split(/[,:]/);
        const first = tokens[0];
        if (first === "g" || first === "b") {
            const count = tokens.length >= 2 ? parseInt(tokens[1]) || 1 : 1;
            const solarTick = tokens.length >= 3 ? parseInt(tokens[2]) || 0 : 0;
            const mcTick = solarTick + 1;
            const mcKey = first === "g" ? "r" : "w";
            for (let i = 0; i < count; i++) {
                result.push(mcTick + ":" + mcKey);
            }
        } else if (first === "t" || first === "p") {
            const solarTick = tokens.length >= 2 ? parseInt(tokens[1]) || 0 : 0;
            const mcTick = solarTick + 1;
            const mcKey = first === "t" ? "t" : "e";
            result.push(mcTick + ":" + mcKey);
        } else {
            const lastColon = line.lastIndexOf(":");
            if (lastColon === -1) {
                result.push("0:" + line);
            } else {
                const maybeTick = line.substring(lastColon + 1);
                if (/^\d+$/.test(maybeTick)) {
                    result.push(maybeTick + ":" + line.substring(0, lastColon));
                } else {
                    result.push("0:" + line);
                }
            }
        }
    }
    return result.join("\n");
}

describe("solar-side vs mclovin-side function parity", () => {
    const solarInputs = [
        "g,1:4", "b,1:9", "t:14", "p:19", "25,17:3",
        "g,1", "g,3:4", "b,2:9",
        "g,1:4\nb,1:9\n25,17:3\nt:14\np:19",
        "t", "p",
    ];

    for (const input of solarInputs) {
        it(`solar→mclovin parity for: ${JSON.stringify(input)}`, () => {
            expect(solar_solarDefCmdsToMclovin(input)).toBe(solarDefenderCommandsToMclovin(input));
        });
    }
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

describe("integration: solar → mclovin parsing", () => {
    it("converted actions parse correctly in mclovin", () => {
        const mc = solarDefenderCommandsToMclovin("g,1:4\nb,1:9\nt:14\np:19");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(5)).toEqual(["r"]);
        expect(parsed!.get(10)).toEqual(["w"]);
        expect(parsed!.get(15)).toEqual(["t"]);
        expect(parsed!.get(20)).toEqual(["e"]);
    });

    it("converted moves parse correctly in mclovin", () => {
        const mc = solarDefenderCommandsToMclovin("25,17:3\n30,22:8");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(3)).toEqual(["25,17"]);
        expect(parsed!.get(8)).toEqual(["30,22"]);
    });

    it("converted no-tick action has tick 1 in mclovin", () => {
        const mc = solarDefenderCommandsToMclovin("g,1");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(1)).toEqual(["r"]);
    });

    it("expanded g,count produces valid mclovin output", () => {
        const mc = solarDefenderCommandsToMclovin("g,3:4");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).not.toBeNull();
        expect(parsed!.get(5)).toEqual(["r", "r", "r"]);
    });

    it("mclovin rejects tick 0 in defender commands", () => {
        const mc = solarDefenderCommandsToMclovin("25,17");
        expect(mc).toBe("0:25,17");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).toBeNull();
    });

    it("out-of-order ticks cause mclovin parse failure", () => {
        const mc = solarDefenderCommandsToMclovin("g,1:10\n25,17:3");
        expect(mc).toBe("11:r\n3:25,17");
        const parsed = mclovinParseDefenderCommands(mc);
        expect(parsed).toBeNull();
    });
});

describe("tick semantics", () => {
    it("action on tick 5: solar WaitUntil=4 → mclovin tick 5", () => {
        expect(solarDefenderCommandsToMclovin("g,1:4")).toBe("5:r");
    });

    it("action on tick 1: solar WaitUntil=0 (omitted) → mclovin tick 1", () => {
        expect(solarDefenderCommandsToMclovin("g,1")).toBe("1:r");
    });

    it("move on tick 5: solar WaitUntil=5 → mclovin tick 5 (no offset)", () => {
        expect(solarDefenderCommandsToMclovin("25,17:5")).toBe("5:25,17");
    });
});
