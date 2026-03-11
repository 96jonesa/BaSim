import {describe, expect, test} from "vitest";
import {HealerCodeAction, parseHealerCodes, assignSpawnPriorities} from "../src/HealerCodeAction";

describe("parseHealerCodes", (): void => {
    test("empty string returns empty array", (): void => {
        expect(parseHealerCodes("")).toEqual([]);
        expect(parseHealerCodes("  ")).toEqual([]);
    });

    test("single healer code", (): void => {
        const actions = parseHealerCodes("h1,3");
        expect(actions.length).toBe(3);
        expect(actions[0].healerId).toBe(1);
        expect(actions[1].healerId).toBe(1);
        expect(actions[2].healerId).toBe(1);
    });

    test("multiple healer codes separated by dash", (): void => {
        const actions = parseHealerCodes("h1,2-h2,3");
        expect(actions.length).toBe(5);
        expect(actions[0].healerId).toBe(1);
        expect(actions[1].healerId).toBe(1);
        expect(actions[2].healerId).toBe(2);
        expect(actions[3].healerId).toBe(2);
        expect(actions[4].healerId).toBe(2);
    });

    test("code with timing on last food", (): void => {
        const actions = parseHealerCodes("h1,3:24");
        expect(actions.length).toBe(3);
        expect(actions[0].waitUntil).toBe(0);
        expect(actions[1].waitUntil).toBe(0);
        expect(actions[2].waitUntil).toBe(24);
    });

    test("single food with timing", (): void => {
        const actions = parseHealerCodes("h2,1:30");
        expect(actions.length).toBe(1);
        expect(actions[0].healerId).toBe(2);
        expect(actions[0].waitUntil).toBe(30);
    });

    test("invalid code throws", (): void => {
        expect(() => parseHealerCodes("invalid")).toThrow("Invalid healer code");
        expect(() => parseHealerCodes("h1")).toThrow("Invalid healer code");
        expect(() => parseHealerCodes("1,3")).toThrow("Invalid healer code");
    });

    test("clone preserves fields", (): void => {
        const action = new HealerCodeAction(2, 15);
        const cloned = action.clone();
        expect(cloned.healerId).toBe(2);
        expect(cloned.waitUntil).toBe(15);
        expect(cloned).not.toBe(action);
    });
});

describe("assignSpawnPriorities", (): void => {
    test("assigns spawn ticks to first poison of each healer", (): void => {
        const actions = parseHealerCodes("h1,2-h2,2");
        assignSpawnPriorities(actions);

        // h1 group first (spawn tick 11), then h2 group (spawn tick 21)
        expect(actions[0].healerId).toBe(1);
        expect(actions[0].waitUntil).toBe(11);
        expect(actions[1].healerId).toBe(1);
        expect(actions[1].waitUntil).toBe(0);

        expect(actions[2].healerId).toBe(2);
        expect(actions[2].waitUntil).toBe(21);
        expect(actions[3].healerId).toBe(2);
        expect(actions[3].waitUntil).toBe(0);
    });

    test("sorts healer groups by spawn tick", (): void => {
        const actions = parseHealerCodes("h3,1-h1,1-h2,1");
        assignSpawnPriorities(actions);

        expect(actions[0].healerId).toBe(1);
        expect(actions[0].waitUntil).toBe(11);

        expect(actions[1].healerId).toBe(2);
        expect(actions[1].waitUntil).toBe(21);

        expect(actions[2].healerId).toBe(3);
        expect(actions[2].waitUntil).toBe(31);
    });

    test("preserves explicit timing over spawn tick", (): void => {
        const actions = parseHealerCodes("h1,1:50");
        assignSpawnPriorities(actions);

        expect(actions[0].healerId).toBe(1);
        expect(actions[0].waitUntil).toBe(50);
    });

    test("repoisons follow first poison in order", (): void => {
        const actions = parseHealerCodes("h1,3");
        assignSpawnPriorities(actions);

        expect(actions[0].waitUntil).toBe(11);
        expect(actions[1].waitUntil).toBe(0);
        expect(actions[2].waitUntil).toBe(0);
        expect(actions[0].healerId).toBe(1);
        expect(actions[1].healerId).toBe(1);
        expect(actions[2].healerId).toBe(1);
    });

    test("merges multiple segments for same healer", (): void => {
        const actions = parseHealerCodes("h1,1-h2,1-h1,1");
        assignSpawnPriorities(actions);

        // h1 group (2 actions) sorted first, h2 group sorted second
        expect(actions[0].healerId).toBe(1);
        expect(actions[0].waitUntil).toBe(11);
        expect(actions[1].healerId).toBe(1);
        expect(actions[1].waitUntil).toBe(0);

        expect(actions[2].healerId).toBe(2);
        expect(actions[2].waitUntil).toBe(21);
    });
});
