import { Command } from "./Command.js";
export class JogreBonesCommand extends Command {
    constructor() {
        super();
    }
    clone() {
        return new JogreBonesCommand();
    }
}
