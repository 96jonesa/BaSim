import {Position} from "./Position.js";
import {RGBColor} from "./RGBColor";

export class TileMarker {
    public position: Position;
    public rgbColor: RGBColor;

    public constructor(position: Position, rgbColor: RGBColor) {
        this.position = position;
        this.rgbColor = rgbColor;
    }
}