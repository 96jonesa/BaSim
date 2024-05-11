export class RGBColor {
    public r: number;
    public g: number;
    public b: number;

    public constructor(r: number, g: number, b: number) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    public static fromHexColor(hexColor: number) {
        return new RGBColor((hexColor >> 16) & 255, (hexColor >> 8) & 255, hexColor & 255);
    }
}