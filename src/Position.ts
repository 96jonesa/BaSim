export class Position {
    public x: number;
    public y: number;

    public constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    public equals(position: Position): boolean {
        return this.x === position.x && this.y === position.y;
    }

    public distance(position: Position): number {
        return Math.max(Math.abs(this.x - position.x), Math.abs(this.y - position.y));
    }

    public closestAdjacentPosition(position: Position): Position {
        const northTileDistance: number = this.distance(new Position(position.x, position.y + 1));
        const southTileDistance: number = this.distance(new Position(position.x, position.y - 1));
        const eastTileDistance: number = this.distance(new Position(position.x + 1, position.y));
        const westTileDistance: number = this.distance(new Position(position.x - 1, position.y));

        const minimumDistance: number = Math.min(northTileDistance, southTileDistance, eastTileDistance, westTileDistance);

        if (minimumDistance === northTileDistance) {
            return new Position(position.x, position.y + 1);
        }

        if (minimumDistance === southTileDistance) {
            return new Position(position.x, position.y - 1);
        }

        if (minimumDistance === eastTileDistance) {
            return new Position(position.x + 1, position.y);
        }

        return new Position(position.x - 1, position.y);
    }

    public clone(): Position {
        let position: Position = new Position(this.x, this.y);
        position.x = this.x;
        position.y = this.y;

        return position;
    }
}