import {describe, expect, test} from "vitest";
import {HealerPenance} from "../src/HealerPenance";
import {Position} from "../src/Position";
import {BarbarianAssault} from "../src/BarbarianAssault";
import {EggQueueItem} from "../src/EggQueueItem";
import {EggType} from "../src/EggType";
import {CannonSide} from "../src/CannonSide";
import {Command} from "../src/Command";

describe("processEggQueue", (): void => {
    test("red egg deals 3 damage when stalled reaches 0", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        healer.processEggQueue(ba);

        expect(healer.health).toBe(24);
        expect(healer.poisonHitsplat).toBe(true);
    });

    test("red egg does not damage zombie healer", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.zombieState = true;
        healer.health = 0;

        healer.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        healer.processEggQueue(ba);

        expect(healer.health).toBe(0);
    });

    test("green egg deals 1 damage and starts poison", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.eggQueue.push(new EggQueueItem(0, EggType.GREEN, CannonSide.WEST));
        healer.processEggQueue(ba);

        expect(healer.health).toBe(26);
        expect(healer.greenCounter).toBe(148);
        expect(healer.poisonHitsplat).toBe(true);
    });

    test("green egg does not damage zombie healer", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.zombieState = true;
        healer.health = 0;

        healer.eggQueue.push(new EggQueueItem(0, EggType.GREEN, CannonSide.WEST));
        healer.processEggQueue(ba);

        expect(healer.health).toBe(0);
        expect(healer.greenCounter).toBe(148);
    });

    test("green poison ticks damage every 30 ticks", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.greenCounter = 120;
        healer.processEggQueue(ba);

        expect(healer.health).toBe(26);
        expect(healer.poisonHitsplat).toBe(true);
    });

    test("green poison does not tick at non-multiple of 30", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.poisonHitsplat = false;

        healer.greenCounter = 25;
        healer.processEggQueue(ba);

        expect(healer.health).toBe(27);
    });

    test("green poison does not damage zombie healer on tick", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.zombieState = true;
        healer.health = 0;

        healer.greenCounter = 30;
        healer.processEggQueue(ba);

        expect(healer.health).toBe(0);
    });

    test("blue egg sets blueCounter and clears queue", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.eggQueue.push(new EggQueueItem(2, EggType.RED, CannonSide.WEST));
        healer.eggQueue.push(new EggQueueItem(0, EggType.BLUE, CannonSide.EAST));
        healer.processEggQueue(ba);

        expect(healer.blueCounter).toBe(9);
        expect(healer.eggQueue.length).toBe(0);
    });

    test("blue egg on dying healer creates zombie at cannon position", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.isDying = true;
        healer.health = 0;

        healer.eggQueue.push(new EggQueueItem(0, EggType.BLUE, CannonSide.WEST));
        healer.processEggQueue(ba);

        expect(healer.isDying).toBe(false);
        expect(healer.zombieState).toBe(true);
        expect(healer.position.x).toBe(21);
        expect(healer.position.y).toBe(26);
        expect(healer.despawnCountdown).toBe(3);
    });

    test("zombie state clears when health becomes positive", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.zombieState = true;
        healer.health = 1;

        healer.processEggQueue(ba);

        expect(healer.zombieState).toBe(false);
    });

    test("zombie state persists when health is 0", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        healer.zombieState = true;
        healer.health = 0;

        healer.processEggQueue(ba);

        expect(healer.zombieState).toBe(true);
    });

    test("stalled counter decrements each call", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.eggQueue.push(new EggQueueItem(4, EggType.RED, CannonSide.WEST));
        healer.processEggQueue(ba);

        expect(healer.health).toBe(27);
        expect(healer.eggQueue[0].stalled).toBe(3);
    });

    test("egg removed from queue after landing", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        healer.processEggQueue(ba);
        healer.processEggQueue(ba);

        expect(healer.eggQueue.length).toBe(0);
    });

    test("blue stun decrements each call", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        healer.blueCounter = 5;
        healer.processEggQueue(ba);

        expect(healer.blueCounter).toBe(4);
    });

    test("healer skips movement when stunned", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        ba.ticks = 10;

        healer.blueCounter = 3;
        const startX = healer.position.x;
        const startY = healer.position.y;

        healer.tick(ba);

        expect(healer.position.x).toBe(startX);
        expect(healer.position.y).toBe(startY);
    });

    test("clone copies egg-related fields", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        healer.blueCounter = 3;
        healer.greenCounter = 50;
        healer.zombieState = true;
        healer.eggQueue.push(new EggQueueItem(4, EggType.GREEN, CannonSide.EAST));

        const cloned = healer.clone();

        expect(cloned.blueCounter).toBe(3);
        expect(cloned.greenCounter).toBe(50);
        expect(cloned.zombieState).toBe(true);
        expect(cloned.eggQueue.length).toBe(1);
        expect(cloned.eggQueue[0]).not.toBe(healer.eggQueue[0]);
        expect(cloned.eggQueue[0].stalled).toBe(4);
        expect(cloned.eggQueue[0].type).toBe(EggType.GREEN);
    });

    test("clone copies forcedTarget", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
        healer.forcedTarget = "mainsecond";

        const cloned = healer.clone();

        expect(cloned.forcedTarget).toBe("mainsecond");
    });

    test("lastPoisonTick initialized to spawnTick", (): void => {
        const healer = new HealerPenance(new Position(42, 37), 27, 11, 1);

        expect(healer.lastPoisonTick).toBe(11);
    });
});
