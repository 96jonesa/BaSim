export class RGBColor {
    constructor(r, g, b) {
        this.r = r;
        this.g = g;
        this.b = b;
    }
    static fromHexColor(hexColor) {
        return new RGBColor((hexColor >> 16) & 255, (hexColor >> 8) & 255, hexColor & 255);
    }
}
