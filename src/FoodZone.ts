import {Food} from "./Food.js";

/**
 * Represents a section of a {@link BarbarianAssaultMap} that contains zero or more {@link Food}.
 */
export class FoodZone {
    public foodList: Array<Food>;
    
    public constructor(foodList: Array<Food>) {
        this.foodList = foodList;
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): FoodZone {
        let foodZone: FoodZone = new FoodZone(this.foodList);
        foodZone.foodList = [];
        for (let i: number = 0; i < this.foodList.length; i++) {
            foodZone.foodList.push(this.foodList[i] === null ? null! : this.foodList[i].clone());
        }

        return foodZone;
    }
}