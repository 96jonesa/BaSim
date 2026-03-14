import {Command} from "./Command.js";
import {SeedType} from "./SeedType.js";

export class SeedCommand extends Command {
    public readonly seedType: SeedType;

    public constructor(seedType: SeedType) {
        super();
        this.seedType = seedType;
    }

    public clone(): SeedCommand {
        return new SeedCommand(this.seedType);
    }
}
