import {describe, expect, test} from "vitest";
import {Food} from "../src/Food.js";
import {FoodType} from "../src/FoodType.js";
import {Position} from "../src/Position.js";
import {FoodZone} from "../src/FoodZone.js";

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const position: Position = null!;
        const foodType: FoodType = FoodType.TOFU;
        const isGood: boolean = true;
        const food: Food = new Food(position, foodType, isGood);
        const foodList: Array<Food> = [food];

        const foodZone: FoodZone = new FoodZone(foodList);
        const foodZoneClone: FoodZone = foodZone.clone();

        expect(foodZoneClone).not.toBe(foodZone);
        expect(JSON.stringify(foodZoneClone)).toBe(JSON.stringify(foodZone));
    });

    test("clone is a deep copy with null food", (): void => {
        const food: Food = null!;
        const foodList: Array<Food> = [food];

        const foodZone: FoodZone = new FoodZone(foodList);
        const foodZoneClone: FoodZone = foodZone.clone();

        expect(foodZoneClone).not.toBe(foodZone);
        expect(JSON.stringify(foodZoneClone)).toBe(JSON.stringify(foodZone));
    });
});