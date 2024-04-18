import {Command} from "./Command.js";
import {DefenderActionType} from "./DefenderActionType.js";

export class DefenderActionCommand extends Command {
    public type: DefenderActionType;

    public constructor(type: DefenderActionType) {
        super();

        this.type = type;
    }

    public clone(): DefenderActionCommand {
        return new DefenderActionCommand(this.type);
    }
}