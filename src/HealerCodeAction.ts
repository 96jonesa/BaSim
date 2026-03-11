export class HealerCodeAction {
    public healerId: number;
    public waitUntil: number;

    public constructor(healerId: number, waitUntil: number) {
        this.healerId = healerId;
        this.waitUntil = waitUntil;
    }

    public clone(): HealerCodeAction {
        return new HealerCodeAction(this.healerId, this.waitUntil);
    }
}

export function parseHealerCodes(input: string): Array<HealerCodeAction> {
    const actions: Array<HealerCodeAction> = [];

    if (input.trim().length === 0) {
        return actions;
    }

    const segments = input.trim().split("-");

    for (const segment of segments) {
        const trimmed = segment.trim();
        if (trimmed.length === 0) {
            continue;
        }

        // Format: h<id>,<count> or h<id>,<count>:<waitUntil>
        const match = trimmed.match(/^h(\d+),(\d+)(?::(\d+))?$/);
        if (match === null) {
            throw new Error("Invalid healer code: " + trimmed);
        }

        const healerId = parseInt(match[1]);
        const count = parseInt(match[2]);
        const waitUntil = match[3] !== undefined ? parseInt(match[3]) : 0;

        for (let i = 0; i < count; i++) {
            if (i === count - 1 && waitUntil > 0) {
                actions.push(new HealerCodeAction(healerId, waitUntil));
            } else {
                actions.push(new HealerCodeAction(healerId, 0));
            }
        }
    }

    return actions;
}

export function assignSpawnPriorities(actions: Array<HealerCodeAction>): void {
    // Group actions by healer ID, preserving order within each group
    const groups = new Map<number, Array<HealerCodeAction>>();
    const groupOrder: Array<number> = [];

    for (const action of actions) {
        if (!groups.has(action.healerId)) {
            groups.set(action.healerId, []);
            groupOrder.push(action.healerId);
        }
        groups.get(action.healerId).push(action);
    }

    // Assign spawn tick to the first action per healer (if not already set)
    for (const [healerId, group] of groups) {
        if (group[0].waitUntil === 0) {
            // Healers spawn at tick 11, 21, 31, etc.
            group[0].waitUntil = healerId * 10 + 1;
        }
    }

    // Sort groups by the first action's waitUntil
    groupOrder.sort((a, b) => groups.get(a)[0].waitUntil - groups.get(b)[0].waitUntil);

    // Flatten back into the actions array
    actions.length = 0;
    for (const healerId of groupOrder) {
        actions.push(...groups.get(healerId));
    }
}
