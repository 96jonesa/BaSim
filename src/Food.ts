import {Position} from "./Position.js";
import {FoodType} from "./FoodType.js";

/**
 * Represents food.
 */
export class Food {
    public position: Position;
    public type: FoodType;
    public isGood: boolean;
    public colorRed: number;
    public colorGreen: number;
    public colorBlue: number = 0;

    public constructor(position: Position, type: FoodType, isGood: boolean) {
        this.position = position;
        this.type = type;
        this.isGood = isGood;

        this.colorRed = isGood ? 0 : 255;
        this.colorGreen = isGood ? 255 : 0;
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): Food {
        let food: Food = new Food(this.position, this.type, this.isGood);
        food.position = this.position === null ? null! : this.position.clone();
        food.type = this.type;
        food.isGood = this.isGood;
        food.colorRed = this.colorRed;
        food.colorGreen = this.colorGreen;
        food.colorBlue = this.colorBlue;

        return food;
    }
}