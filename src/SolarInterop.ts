/**
 * Conversion functions for importing Solar BA sim settings into McLovin sim.
 *
 * McLovin format: tick:command (e.g. "5:20,21" or "3:r")
 * Solar format: command:tick (e.g. "20,21:5" or "g,1:3")
 *
 * Solar defender uses different command letters and a WaitUntil system where
 * action commands execute when TickCounter > WaitUntil (one tick after the value),
 * while move commands execute when TickCounter >= WaitUntil (on that tick).
 */

/**
 * Converts solar command format (x,y:tick or cmd:tick) to mclovin format (tick:x,y or tick:cmd).
 * Lines without a tick suffix are treated as tick 0 commands.
 */
export function solarCommandsToMclovin(commands: string): string {
    return commands.split("\n").map(line => {
        line = line.trim();
        if (line.length === 0) return "";
        const lastColon = line.lastIndexOf(":");
        if (lastColon === -1) return "0:" + line;
        const maybeTick = line.substring(lastColon + 1);
        if (/^\d+$/.test(maybeTick)) {
            const rest = line.substring(0, lastColon);
            return maybeTick + ":" + rest;
        }
        return "0:" + line;
    }).join("\n");
}

/**
 * Converts solar defender commands to mclovin defender format.
 * g,count→count r lines  b,count→count w lines  t→t  p→e  x,y→x,y
 * Solar defender actions use WaitUntil with a >= check (executes when tick > WaitUntil),
 * so action ticks need to be offset by +1 compared to solar's tick.
 * Move commands use a <= check and don't need offset.
 */
export function solarDefenderCommandsToMclovin(commands: string): string {
    const result: string[] = [];
    for (let line of commands.split("\n")) {
        line = line.trim();
        if (line.length === 0) continue;
        const tokens = line.split(/[,:]/);
        const first = tokens[0];
        if (first === "g" || first === "b") {
            const count = tokens.length >= 2 ? parseInt(tokens[1]) || 1 : 1;
            const solarTick = tokens.length >= 3 ? parseInt(tokens[2]) || 0 : 0;
            if (solarTick <= 1) continue;
            const mcKey = first === "g" ? "r" : "w";
            for (let i = 0; i < count; i++) {
                result.push(solarTick + ":" + mcKey);
            }
        } else if (first === "t" || first === "p") {
            const solarTick = tokens.length >= 2 ? parseInt(tokens[1]) || 0 : 0;
            if (solarTick <= 1) continue;
            const mcKey = first === "t" ? "t" : "e";
            result.push(solarTick + ":" + mcKey);
        } else {
            const lastColon = line.lastIndexOf(":");
            if (lastColon === -1) {
                continue;
            } else {
                const maybeTick = line.substring(lastColon + 1);
                if (/^\d+$/.test(maybeTick)) {
                    const solarTick = parseInt(maybeTick);
                    if (solarTick <= 1) continue;
                    result.push((solarTick - 1) + ":" + line.substring(0, lastColon));
                } else {
                    result.push("0:" + line);
                }
            }
        }
    }
    return result.join("\n");
}

/**
 * Splits solar's combined healer spawns format ("11:mc,21:h,31") into mclovin's
 * separate healer spawns and healer spawn targets.
 */
export function solarHealerSpawnsToMclovin(combined: string): {spawns: string, targets: string} {
    const parts = combined.split(/[,\-]/).map(s => s.trim()).filter(s => s.length > 0);
    const spawns: string[] = [];
    const targets: string[] = [];
    let hasAnyTarget = false;
    for (const part of parts) {
        const colonIdx = part.indexOf(":");
        if (colonIdx !== -1) {
            spawns.push(part.substring(0, colonIdx));
            targets.push(part.substring(colonIdx + 1));
            hasAnyTarget = true;
        } else {
            spawns.push(part);
            targets.push("");
        }
    }
    // Trim trailing empty targets
    while (targets.length > 0 && targets[targets.length - 1] === "") {
        targets.pop();
    }
    return {
        spawns: spawns.join(","),
        targets: hasAnyTarget ? targets.join("-") : ""
    };
}
