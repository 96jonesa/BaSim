import { CannonSide } from "./CannonSide.js";
import { CannonCommand } from "./CannonCommand.js";
import { PenanceType } from "./PenanceType.js";
import { EggType } from "./EggType.js";
import { EggQueueItem } from "./EggQueueItem.js";
import { CANNON_RANGE, getCannonPosition } from "./CannonPositions.js";
export { getCannonPosition } from "./CannonPositions.js";
export class Cannon {
    constructor() {
        this.queue = [];
        this.npcZones = new Map();
        this.npcZonesHealer = new Map();
    }
    updateZonePriority(npcs, zoneMap) {
        for (const npc of npcs) {
            const newZone = [npc.position.x >>> 3, npc.position.y >>> 3];
            const existing = zoneMap.get(npc.id);
            if (existing === undefined || existing.zone[0] !== newZone[0] || existing.zone[1] !== newZone[1]) {
                zoneMap.set(npc.id, { zone: newZone, counter: 1 });
            }
            else {
                existing.counter++;
            }
        }
    }
    getTarget(cmd, barbarianAssault) {
        const cannonPos = getCannonPosition(cmd.cannon);
        const zoneMap = cmd.penance === PenanceType.RUNNER ? this.npcZones : this.npcZonesHealer;
        let candidates;
        if (cmd.penance === PenanceType.RUNNER) {
            candidates = barbarianAssault.runners
                .filter((r) => {
                if (r.isDying)
                    return false;
                if (r.blueCounter >= 0)
                    return false;
                const dist = cannonPos.distance(r.position);
                return dist <= CANNON_RANGE;
            })
                .map((r) => {
                const zd = zoneMap.get(r.id) || { zone: [r.position.x >>> 3, r.position.y >>> 3], counter: 1 };
                return {
                    npc: r,
                    dist: cannonPos.euclideanDistance(r.position),
                    zone: zd.zone,
                    zoneCounter: zd.counter,
                };
            });
        }
        else {
            candidates = barbarianAssault.healers
                .filter((h) => {
                if (h.isDying)
                    return false;
                if (h.blueCounter >= 0)
                    return false;
                const dist = cannonPos.distance(h.position);
                return dist <= CANNON_RANGE;
            })
                .map((h) => {
                const zd = zoneMap.get(h.id) || { zone: [h.position.x >>> 3, h.position.y >>> 3], counter: 1 };
                return {
                    npc: h,
                    dist: cannonPos.euclideanDistance(h.position),
                    zone: zd.zone,
                    zoneCounter: zd.counter,
                };
            });
        }
        if (candidates.length === 0)
            return null;
        // Sort by: distance (asc), zone x (asc), zone y (asc), zone counter (asc), id (asc)
        candidates.sort((a, b) => {
            if (a.dist !== b.dist)
                return a.dist - b.dist;
            if (a.zone[0] !== b.zone[0])
                return a.zone[0] - b.zone[0];
            if (a.zone[1] !== b.zone[1])
                return a.zone[1] - b.zone[1];
            if (a.zoneCounter !== b.zoneCounter)
                return a.zoneCounter - b.zoneCounter;
            return a.npc.penanceId - b.npc.penanceId;
        });
        return candidates[0].npc;
    }
    calculateTravelTime(cannonPos, targetPos, eggType) {
        const dist = cannonPos.distance(targetPos);
        if (dist > CANNON_RANGE)
            return -1;
        let travelTime;
        if (dist > 9) {
            travelTime = 6;
        }
        else if (dist > 3) {
            travelTime = 5;
        }
        else {
            travelTime = 4;
        }
        if (eggType === EggType.RED) {
            travelTime += 1;
        }
        return travelTime;
    }
    tick(barbarianAssault) {
        this.updateZonePriority(barbarianAssault.runners, this.npcZones);
        this.updateZonePriority(barbarianAssault.healers, this.npcZonesHealer);
        for (const cmd of this.queue) {
            if (cmd.numEggs <= 0)
                continue;
            if (cmd.tick > barbarianAssault.ticks)
                continue;
            if (cmd.stalled > 0) {
                cmd.stalled--;
                continue;
            }
            const target = this.getTarget(cmd, barbarianAssault);
            if (target === null)
                continue;
            const cannonPos = getCannonPosition(cmd.cannon);
            const travelTime = this.calculateTravelTime(cannonPos, target.position, cmd.eggType);
            if (travelTime < 0)
                continue;
            const egg = new EggQueueItem(travelTime, cmd.eggType, cmd.cannon);
            target.eggQueue.push(egg);
            cmd.numEggs--;
            cmd.stalled = travelTime;
        }
    }
    clone() {
        const cannon = new Cannon();
        cannon.queue = this.queue.map(cmd => cmd.clone());
        cannon.npcZones = new Map();
        this.npcZones.forEach((val, key) => {
            cannon.npcZones.set(key, { zone: [val.zone[0], val.zone[1]], counter: val.counter });
        });
        cannon.npcZonesHealer = new Map();
        this.npcZonesHealer.forEach((val, key) => {
            cannon.npcZonesHealer.set(key, { zone: [val.zone[0], val.zone[1]], counter: val.counter });
        });
        return cannon;
    }
}
export function parseCannonInput(input, secondsMode = false) {
    if (input.trim() === "")
        return [];
    const commands = [];
    const parts = input.split("-");
    let id = 0;
    let lastCannon = CannonSide.WEST;
    let lastPenance = PenanceType.RUNNER;
    let lastEggType = EggType.RED;
    for (const part of parts) {
        const tokens = part.trim().split(",");
        if (tokens.length === 5) {
            // Full format: cannon+penance+eggType, numEggs, tick
            // e.g. wrr,1,51 but split as: w, r, r, 1, 51
            // Actually the format is: "wrr,1,51" = "cannonSide penanceType eggType, numEggs, tick"
            // But it's 3 chars then comma-separated: wrr is one token
            // Let me re-read: the format from solar is "wrr,1,51" where wrr = west+runner+red
            // So token[0] = "wrr", token[1] = "1", token[2] = "51"
            // That's only 3 tokens when split by comma. Let me fix this.
        }
        function parseTick(value) {
            if (secondsMode) {
                const seconds = parseFloat(value.trim());
                if (isNaN(seconds))
                    return NaN;
                const t = seconds / 0.6 + 1;
                const rounded = Math.round(t);
                if (Math.abs(t - rounded) > 0.001 || rounded < 1)
                    return NaN;
                return rounded;
            }
            return parseInt(value.trim());
        }
        if (tokens.length === 3) {
            // Full format: "wrr,1,51"
            const descriptor = tokens[0].trim();
            const numEggs = parseInt(tokens[1].trim());
            const tick = parseTick(tokens[2]);
            if (descriptor.length === 3) {
                const cannonChar = descriptor[0];
                const penanceChar = descriptor[1];
                const eggChar = descriptor[2];
                lastCannon = cannonChar === "e" ? CannonSide.EAST : CannonSide.WEST;
                lastPenance = penanceChar === "h" ? PenanceType.HEALER : PenanceType.RUNNER;
                lastEggType = eggChar === "g" ? EggType.GREEN : eggChar === "b" ? EggType.BLUE : EggType.RED;
            }
            if (isNaN(numEggs) || isNaN(tick))
                return null;
            commands.push(new CannonCommand(id++, lastCannon, lastPenance, lastEggType, numEggs, tick));
        }
        else if (tokens.length === 2) {
            // Shorthand: "1,51" - inherit cannon/penance/eggType from last
            const numEggs = parseInt(tokens[0].trim());
            const tick = parseTick(tokens[1]);
            if (isNaN(numEggs) || isNaN(tick))
                return null;
            commands.push(new CannonCommand(id++, lastCannon, lastPenance, lastEggType, numEggs, tick));
        }
        else {
            return null;
        }
    }
    return commands;
}
