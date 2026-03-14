import { Command } from "./Command.js";
export class SeedCommand extends Command {
    constructor(seedType) {
        super();
        this.seedType = seedType;
    }
    clone() {
        return new SeedCommand(this.seedType);
    }
}
