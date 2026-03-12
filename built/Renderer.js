const PIXEL_ALPHA = 255 << 24;
/**
 * Utility class for rendering image details.
 */
export class Renderer {
    constructor(canvas, width, height, tileSize, tileOffsetX = 0) {
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
    get visibleTilesX() {
        return this.canvasWidth / this.tileSize;
    }
    isTileVisibleX(x) {
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
    fillOpaque(x, y) {
        if (!this.isTileVisibleX(x))
            return;
        this.setFilledRectangle((x - this.tileOffsetX) * this.tileSize, y * this.tileSize, this.tileSize, this.tileSize);
    }
    /**
     * Draws over every pixel in the tile specified by the given tile coordinates, with the current
     * draw color and draw opacity (leaving existing drawn colors in place beneath the new drawing).
     *
     * @param x the x coordinate of the tile to draw over the pixels of
     * @param y the y coordinate of the tile to draw over the pixels of
     */
    fill(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    outline(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    outlineBig(x, y, width, height) {
        let startX = x - this.tileOffsetX;
        let w = width;
        if (startX < 0) {
            w += startX;
            startX = 0;
        }
        if (startX + w > this.visibleTilesX) {
            w = this.visibleTilesX - startX;
        }
        if (w <= 0)
            return;
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
    westLine(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    eastLine(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    eastLineBig(x, y, length) {
        if (!this.isTileVisibleX(x))
            return;
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
    southLine(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    northLine(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    northLineBig(x, y, length) {
        let startX = x - this.tileOffsetX;
        let len = length;
        if (startX < 0) {
            len += startX;
            startX = 0;
        }
        if (startX + len > this.visibleTilesX) {
            len = this.visibleTilesX - startX;
        }
        if (len <= 0)
            return;
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
    cone(x, y) {
        if (!this.isTileVisibleX(x))
            return;
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
    fillItem(x, y) {
        if (!this.isTileVisibleX(x))
            return;
        const padding = this.tileSize >>> 2;
        const size = this.tileSize - 2 * padding;
        this.drawFilledRectangle((x - this.tileOffsetX) * this.tileSize + padding, y * this.tileSize + padding, size, size);
    }
    /**
     * Resizes the canvas to have the given width and height.
     *
     * @param width     the number of pixels to set the canvas' width to
     * @param height    the number of pixels to set the canvas' height to
     * @private
     */
    resizeCanvas(width, height) {
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
    setDrawColor(r, g, b, a) {
        this.drawColorRB = r | (b << 16);
        this.drawColorG = PIXEL_ALPHA | (g << 8);
        this.drawColor = this.drawColorRB | this.drawColorG;
        this.drawColorA = a + 1;
    }
    /**
     * Draws every pixel with the current draw color and no transparency.
     */
    clear() {
        for (let i = 0; i < this.pixels32.length; i++) {
            this.pixels32[i] = this.drawColor;
        }
    }
    /**
     * Updates the canvas according to drawn pixels.
     */
    present() {
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
    drawPixel(index) {
        const color = this.pixels32[index];
        const oldRB = color & 0xFF00FF;
        const oldAG = color & 0xFF00FF00;
        const rb = oldRB + (this.drawColorA * (this.drawColorRB - oldRB) >> 8) & 0xFF00FF;
        const g = oldAG + (this.drawColorA * (this.drawColorG - oldAG) >> 8) & 0xFF00FF00;
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
    drawHorizontalLine(x, y, length) {
        let index = this.xyToIndex(x, y);
        const endIndex = index + length;
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
    drawVerticalLine(x, y, length) {
        let index = this.xyToIndex(x, y);
        const endIndex = index - length * this.canvasWidth;
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
    setFilledRectangle(x, y, width, height) {
        let index = this.xyToIndex(x, y);
        const rowDelta = width + this.canvasWidth;
        const endYIndex = index - height * this.canvasWidth;
        while (index > endYIndex) {
            const endXIndex = index + width;
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
    drawFilledRectangle(x, y, width, height) {
        let index = this.xyToIndex(x, y);
        const rowDelta = width + this.canvasWidth;
        const endYIndex = index - height * this.canvasWidth;
        while (index > endYIndex) {
            const endXIndex = index + width;
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
    drawOutlinedRectangle(x, y, width, height) {
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
    drawCone(x, y, width) {
        const lastX = x + width - 1;
        const endIndex = (width >>> 1) + (width & 1);
        for (let i = 0; i < endIndex; i++) {
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
    xyToIndex(x, y) {
        return this.canvasYFixOffset + x - y * this.canvasWidth;
    }
}
