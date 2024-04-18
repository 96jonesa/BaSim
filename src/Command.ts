export abstract class Command {
    protected constructor() {
    }

    public abstract clone(): Command;
}