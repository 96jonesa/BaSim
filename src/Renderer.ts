const PIXEL_ALPHA: number = 255 << 24;

/**
 * Utility class for rendering image details.
 */
export class Renderer {
    public canvas: HTMLCanvasElement;
    public context: CanvasRenderingContext2D;
    public canvasWidth: number;
    public canvasHeight: number;
    public canvasYFixOffset: number;
    public imageData: ImageData;
    public pixels: ArrayBuffer;
    public pixels8: Uint8ClampedArray;
    public pixels32: Uint32Array;
    public drawColorRB: number;
    public drawColorG: number;
    public drawColor: number;
    public drawColorA: number;
    public tileSize: number;
    public tileOffsetX: number;

    public constructor(canvas: HTMLCanvasElement, width: number, height: number, tileSize: number, tileOffsetX: number = 0) {
        this.canvas = canvas;
        this.context = canvas.getContext("2d");
        this.tileSize = tileSize;
        this.tileOffsetX = tileOffsetX;

        this.resizeCanvas(width, height);
        this.setDrawColor(255, 255, 255, 255);
    }

    /**
     * Returns the number of visible tiles in the x direction.
     */
    public get visibleTilesX(): number {
        return this.canvasWidth / this.tileSize;
    }

    private isTileVisibleX(x: number): boolean {
        const px = x - this.tileOffsetX;
        return px >= 0 && px < this.visibleTilesX;
    }

    /**
     * Draws every pixel in the tile specified by the given tile coordinates, with the current
     * draw color and no transparency.
     *
     * @param x the x coordinate of the tile to draw the pixels of
     * @param y the y coordinate of the tile to draw the pixels of
     */
    public fillOpaque(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.setFilledRectangle((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the tile specified by the given tile coordinates, with the current
     * draw color and draw opacity (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the pixels of
     * @param y the y coordinate of the tile to draw over the pixels of
     */
    public fill(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawFilledRectangle((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the perimeter (outline) of the tile specified by the given tile
     * coordinates, with the current draw color and draw opacity (leaving existing drawn colors
     * in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the pixels of the perimeter of
     * @param y the y coordinate of the tile to draw over the pixels of the perimeter of
     */
    public outline(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawOutlinedRectangle((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the perimeter (outline) of the rectangle specified by the given
     * upper-left corner tile coordinates, width, and height; with the current draw color and
     * draw opacity (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x         the x coordinate of the upper-left corner tile of the rectangle to draw over
     *                  the pixels of the perimeter of
     * @param y         the y coordinate of the upper-left corner tile of the rectangle to draw over
     *                  the pixels of the perimeter of
     * @param width     the width (in tiles) of the rectangle to draw over the pixels of the
     *                  perimeter of
     * @param height    the height (in tiles) of the rectangle to draw over the pixels of the
     *                  perimeter of
     */
    public outlineBig(x: number, y: number, width: number, height: number): void {
        let startX = x - this.tileOffsetX;
        let w = width;
        if (startX < 0) {
            w += startX;
            startX = 0;
        }
        if (startX + w > this.visibleTilesX) {
            w = this.visibleTilesX - startX;
        }
        if (w <= 0) return;
        this.drawOutlinedRectangle(startX * this.tileSize, y * this.tileSize, this.tileSize * w, this.tileSize * height);
    }

    /**
     * Draws over every pixel in the left vertical line of the perimeter (outline) of the tile
     * specified by the given tile coordinates, with the current draw color and draw opacity
     * (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the left vertical line of the perimeter of
     * @param y the y coordinate of the tile to draw over the left vertical line of the perimeter of
     */
    public westLine(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawVerticalLine((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the right vertical line of the perimeter (outline) of the tile
     * specified by the given tile coordinates, with the current draw color and draw opacity
     * (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the right vertical line of the perimeter of
     * @param y the y coordinate of the tile to draw over the right vertical line of the perimeter of
     */
    public eastLine(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawVerticalLine((x + 1 - this.tileOffsetX) * this.tileSize - 1, y * this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the right vertical line of the perimeter (outline) of the rectangle
     * specified by the given upper-left corner tile coordinates, width, and height; with the current
     * draw color and draw opacity (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the upper-left corner tile of the rectangle to draw over the right
     *          vertical line of the perimeter of
     * @param y the y coordinate of the upper-left corner tile of the rectangle to draw over the right
     *          vertical line of the perimeter of
     */
    public eastLineBig(x: number, y: number, length: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawVerticalLine((x + 1 - this.tileOffsetX) * this.tileSize - 1, y * this.tileSize, this.tileSize * length);
    }

    /**
     * Draws over every pixel in the lower horizontal line of the perimeter (outline) of the tile
     * specified by the given tile coordinates, with the current draw color and draw opacity
     * (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the lower horizontal line of the perimeter of
     * @param y the y coordinate of the tile to draw over the lower horizontal line of the perimeter of
     */
    public southLine(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawHorizontalLine((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the upper horizontal line of the perimeter (outline) of the tile
     * specified by the given tile coordinates, with the current draw color and draw opacity
     * (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the upper horizontal line of the perimeter of
     * @param y the y coordinate of the tile to draw over the upper horizontal line of the perimeter of
     */
    public northLine(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawHorizontalLine((x - this.tileOffsetX) * this.tileSize, (y + 1) * this.tileSize - 1, this.tileSize);
    }

    /**
     * Draws over every pixel in the upper horizontal line of the perimeter (outline) of the rectangle
     * specified by the given upper-left corner tile coordinates, width, and height; with the current
     * draw color and draw opacity (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the upper-left corner tile of the rectangle to draw over the upper
     *          horizontal line of the perimeter of
     * @param y the y coordinate of the upper-left corner tile of the rectangle to draw over the upper
     *          horizontal line of the perimeter of
     */
    public northLineBig(x: number, y: number, length: number): void {
        let startX = x - this.tileOffsetX;
        let len = length;
        if (startX < 0) {
            len += startX;
            startX = 0;
        }
        if (startX + len > this.visibleTilesX) {
            len = this.visibleTilesX - startX;
        }
        if (len <= 0) return;
        this.drawHorizontalLine(startX * this.tileSize, (y + 1) * this.tileSize - 1, this.tileSize * len);
    }

    /**
     * Draws over every pixel in the perimeter of the 2-dimensional cone (isosceles triangle
     * with vertex shared by equal length sides at the center top, and excluding the bottom
     * side) that is one tile wide and half of a tile high, positioned in the bottom half of
     * the tile with given tile coordinates, with the current draw color and draw opacity
     * (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the lower-left-most pixel of the cone to draw over the
     *          pixels of the perimeter of
     * @param y the y coordinate of the lower-left-most pixel of the cone to draw over the
     *          pixels of the perimeter of
     */
    public cone(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        this.drawCone((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize);
    }

    /**
     * Draws over every pixel in the half-size center square of the tile specified by the given tile
     * coordinates, with the current draw color and draw opacity (leaving existing drawn colors in
     * place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the pixels of the half-size center square of
     * @param y the y coordinate of the tile to draw over the pixels of the half-size center square of
     */
    public fillItem(x: number, y: number): void {
        if (!this.isTileVisibleX(x)) return;
        const padding: number = this.tileSize >>> 2;
        const size: number = this.tileSize - 2 * padding;

        this.drawFilledRectangle((x - this.tileOffsetX) * this.tileSize + padding, y * this.tileSize + padding, size, size);
    }

    /**
     * Resizes the canvas to have the given width and height.
     *
     * @param width     the number of pixels to set the canvas' width to
     * @param height    the number of pixels to set the canvas' height to
     * @private
     */
    private resizeCanvas(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvasWidth = width;
        this.canvasHeight = height;
        this.canvasYFixOffset = (height - 1) * width;
        this.imageData = this.context.createImageData(width, height);
        this.pixels = new ArrayBuffer(this.imageData.data.length);
        this.pixels8 = new Uint8ClampedArray(this.pixels);
        this.pixels32 = new Uint32Array(this.pixels);
    }

    /**
     * Sets the draw color to the color specified by the given RGB values, and sets the draw
     * opacity to the given opacity value.
     *
     * @param r the value to set the draw color's R value (of its RGB values) to
     * @param g the value to set the draw color's G value (of its RGB values) to
     * @param b the value to set the draw color's B value (of its RGB values) to
     * @param a the value to set the draw opacity to
     */
    public setDrawColor(r: number, g: number, b: number, a: number): void {
        this.drawColorRB = r | (b << 16);
        this.drawColorG = PIXEL_ALPHA | (g << 8);
        this.drawColor = this.drawColorRB | this.drawColorG;
        this.drawColorA = a + 1;
    }

    /**
     * Draws every pixel with the current draw color and no transparency.
     */
    public clear(): void {
        for (let i: number = 0; i < this.pixels32.length; i++) {
            this.pixels32[i] = this.drawColor;
        }
    }

    /**
     * Updates the canvas according to drawn pixels.
     */
    public present(): void {
        this.imageData.data.set(this.pixels8);
        this.context.putImageData(this.imageData, 0, 0);
    }

    /**
     * Draws over the pixel specified by the given index, with the current draw color and
     * draw opacity.
     *
     * @param index the index of the pixel to draw over
     * @private
     */
    private drawPixel(index: number): void {
        const color: number = this.pixels32[index];
        const oldRB: number = color & 0xFF00FF;
        const oldAG: number = color & 0xFF00FF00;
        const rb: number = oldRB + (this.drawColorA * (this.drawColorRB - oldRB) >> 8) & 0xFF00FF;
        const g: number = oldAG + (this.drawColorA * (this.drawColorG - oldAG) >> 8) & 0xFF00FF00;
        this.pixels32[index] = rb | g;
    }

    /**
     * Draws over every pixel in the horizontal line specified by the given left-most pixel
     * coordinates and length, with the current draw color and draw opacity.
     *
     * @param x         the x coordinate of the left-most pixel of the horizontal line to draw
     *                  over the pixels of
     * @param y         the y coordinate of the left-most pixel of the horizontal line to draw
     *                  over the pixels of
     * @param length    the length (in pixels) of the horizontal line to draw over the pixels of
     * @private
     */
    private drawHorizontalLine(x: number, y: number, length: number): void {
        let index: number = this.xyToIndex(x, y);
        const endIndex: number = index + length;

        for (; index < endIndex; index++) {
            this.drawPixel(index);
        }
    }

    /**
     * Draws over every pixel in the vertical line specified by the given upper-most pixel
     * coordinates and length, with the current draw color and draw opacity.
     *
     * @param x         the x coordinate of the upper-most pixel of the vertical line to draw
     *                  over the pixels of
     * @param y         the y coordinate of the upper-most pixel of the vertical line to draw
     *                  over the pixels of
     * @param length    the length (in pixels) of the vertical line to draw over the pixels of
     * @private
     */
    private drawVerticalLine(x: number, y: number, length: number): void {
        let index: number = this.xyToIndex(x, y);
        const endIndex: number = index - length * this.canvasWidth;

        for (; index > endIndex; index -= this.canvasWidth) {
            this.drawPixel(index);
        }
    }

    /**
     * Draws every pixel in the rectangle specified by the given upper-left corner pixel
     * coordinates, width, and height; with the current draw color and no transparency.
     *
     * @param x         the x coordinate of the upper-left corner of the rectangle to draw the
     *                  pixels of
     * @param y         the y coordinate of the upper-left corner of the rectangle to draw the
     *                  pixels of
     * @param width     the width (in pixels) of the rectangle to draw the pixels of
     * @param height    the height (in pixels) of the rectangle to draw the pixels of
     * @private
     */
    private setFilledRectangle(x: number, y: number, width: number, height: number): void {
        let index: number = this.xyToIndex(x, y);
        const rowDelta: number = width + this.canvasWidth;
        const endYIndex: number = index - height * this.canvasWidth;

        while (index > endYIndex) {
            const endXIndex: number = index + width;

            for (; index < endXIndex; index++) {
                this.pixels32[index] = this.drawColor;
            }

            index -= rowDelta;
        }
    }

    /**
     * Draws over every pixel in the rectangle specified by the given upper-left corner pixel
     * coordinates, width, and height; with the current draw color and draw opacity (leaving
     * existing drawn colors in place beneath the new drawing).
     *
     * @param x         the x coordinate of the upper-left corner of the rectangle to draw over
     *                  the pixels of
     * @param y         the y coordinate of the upper-left corner of the rectangle to draw over
     *                  the pixels of
     * @param width     the width (in pixels) of the rectangle to draw over the pixels of
     * @param height    the height (in pixels) of the rectangle to draw over the pixels of
     * @private
     */
    private drawFilledRectangle(x: number, y: number, width: number, height: number): void {
        let index: number = this.xyToIndex(x, y);
        const rowDelta: number = width + this.canvasWidth;
        const endYIndex: number = index - height * this.canvasWidth;

        while (index > endYIndex) {
            const endXIndex: number = index + width;

            for (; index < endXIndex; index++) {
               this.drawPixel(index);
            }

            index -= rowDelta;
        }
    }

    /**
     * Draws over every pixel in the perimeter (outline) of the rectangle specified by the given
     * upper-left corner pixel coordinates, width, and height; with the current draw color and
     * draw opacity (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x         the x coordinate of the upper-left corner of the rectangle to draw over
     *                  the pixels of the perimeter of
     * @param y         the y coordinate of the upper-left corner of the rectangle to draw over
     *                  the pixels of the perimeter of
     * @param width     the width (in pixels) of the rectangle to draw over the pixels of the
     *                  perimeter of
     * @param height    the height (in pixels) of the rectangle to draw over the pixels of the
     *                  perimeter of
     * @private
     */
    private drawOutlinedRectangle(x: number, y: number, width: number, height: number): void {
        this.drawHorizontalLine(x, y, width);
        this.drawHorizontalLine(x, y + height - 1, width);
        this.drawVerticalLine(x, y + 1, height - 2);
        this.drawVerticalLine(x + width - 1, y + 1, height - 2);
    }

    /**
     * Draws over every pixel in the perimeter of the 2-dimensional cone (isosceles triangle
     * with vertex shared by equal length sides at the center top, and excluding the bottom
     * side) specified by the given pixel coordinates of the lower-left-most pixel of the cone
     * and width (length of the bottom side), with the current draw color and draw opacity
     * (leaving existing drawn colors in place beneath the new drawing). The height of the cone
     * is half of its width.
     *
     * @param x     the x coordinate of the lower-left-most pixel of the cone to draw over the
     *              pixels of the perimeter of
     * @param y     the y coordinate of the lower-left-most pixel of the cone to draw over the
     *              pixels of the perimeter of
     * @param width the width (length of bottom side) (in pixels) of the cone to draw over the
     *              pixels of the perimeter of
     * @private
     */
    private drawCone(x: number, y: number, width: number): void {
        const lastX: number = x + width - 1;
        const endIndex = (width >>> 1) + (width & 1);

        for (let i: number = 0; i < endIndex; i++) {
            this.drawPixel(this.xyToIndex(x + i, y));
            this.drawPixel(this.xyToIndex(lastX - i, y));

            y++;
        }
    }

    /**
     * Gets the index corresponding to the given pixel coordinates.
     *
     * @param x the x coordinate of the pixel coordinates to get the index of
     * @param y the y coordinate of the pixel coordinates to get the index of
     * @return  the index corresponding to the given pixel coordinates
     * @private
     */
    private xyToIndex(x: number, y: number): number {
        return this.canvasYFixOffset + x - y * this.canvasWidth;
    }
}