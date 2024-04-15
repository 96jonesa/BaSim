import {describe, expect, MockInstance, test, vi} from "vitest";
import {RunnerPenance} from "../src/RunnerPenance";
import {Position} from "../src/Position";
import {RunnerPenanceRng} from "../src/RunnerPenanceRng";
import {BarbarianAssault} from "../src/BarbarianAssault";

describe("tick", (): void => {
    test("cycleTick increases by 1 when less than 10", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.cycleTick = 9;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.cycleTick).toBe(10);
    });

    test("cycleTick resets to 1 after 10", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.cycleTick = 10;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.cycleTick).toBe(1);
    });

    test("ticksStandingStill increases by 1 when position is unchanged", (): void => {

    });

    test("despawnCountdown decreases by 1 when not null", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 9;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.despawnCountdown).toBe(8);
    });

    test("added to runnersToRemove when despawnCountdown reaches 0 (1 at start of tick)", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.despawnCountdown).toBe(0);

        expect(barbarianAssault.runnersToRemove.indexOf(runnerPenance)).not.toBe(-1);
    });

    test("runnersAlive decreases by 1 when not dying and despawnCountdown reaches 0 (1 at start of tick)", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = false;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        barbarianAssault.runnersAlive = 1;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.runnersAlive).toBe(0);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("eastTrapCharges decreases by 1 when positive and dying %s east trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        barbarianAssault.eastTrapCharges = 2;

        const position: Position = new Position(barbarianAssault.eastTrapPosition.x + relativeX, barbarianAssault.eastTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.eastTrapCharges).toBe(1);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("eastTrapCharges remains 0 when 0 and dying %s east trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        barbarianAssault.eastTrapCharges = 0;

        const position: Position = new Position(barbarianAssault.eastTrapPosition.x + relativeX, barbarianAssault.eastTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.eastTrapCharges).toBe(0);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("westTrapCharges decreases by 1 when positive and dying %s west trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        barbarianAssault.westTrapCharges = 2;

        const position: Position = new Position(barbarianAssault.westTrapPosition.x + relativeX, barbarianAssault.westTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.westTrapCharges).toBe(1);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("westTrapCharges remains 0 when 0 and dying %s west trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(1, true, true, false, [], 5);

        barbarianAssault.westTrapCharges = 0;

        const position: Position = new Position(barbarianAssault.westTrapPosition.x + relativeX, barbarianAssault.westTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.westTrapCharges).toBe(0);
    });
});