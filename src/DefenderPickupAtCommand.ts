import {Command} from "./Command.js";
import {Position} from "./Position.js";

export class DefenderPickupAtCommand extends Command {
    public readonly position: Position;

    public constructor(position: Position) {
        super();
        this.position = position;
    }

    public clone(): DefenderPickupAtCommand {
        return new DefenderPickupAtCommand(this.position.clone());
    }
}
