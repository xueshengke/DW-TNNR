%% read mask images and check them
close all; clear; clc;

block_circle = imread('block_circle.bmp');
block_circle(block_circle < 128) = 0;
block_circle(block_circle >= 128) = 255;
figure(1), imshow(block_circle, 'border', 'tight')
unique(block_circle)
imwrite(block_circle, 'block_circle.bmp', 'bmp');

block_circle_small = imread('block_circle_small.bmp');
block_circle_small(block_circle_small < 128) = 0;
block_circle_small(block_circle_small >= 128) = 255;
figure(1), imshow(block_circle_small, 'border', 'tight')
unique(block_circle_small)
imwrite(block_circle_small, 'block_circle_small.bmp', 'bmp');

block_diamond = imread('block_diamond.bmp');
block_diamond(block_diamond < 128) = 0;
block_diamond(block_diamond >= 128) = 255;
figure(1), imshow(block_diamond, 'border', 'tight')
unique(block_diamond)
imwrite(block_diamond, 'block_diamond.bmp', 'bmp');

block_diamond_small = imread('block_diamond_small.bmp');
block_diamond_small(block_diamond_small < 128) = 0;
block_diamond_small(block_diamond_small >= 128) = 255;
figure(1), imshow(block_diamond_small, 'border', 'tight')
unique(block_diamond_small)
imwrite(block_diamond_small, 'block_diamond_small.bmp', 'bmp');

block_square = imread('block_square.bmp');
block_square(block_square < 128) = 0;
block_square(block_square >= 128) = 255;
figure(1), imshow(block_square, 'border', 'tight')
unique(block_square)
imwrite(block_square, 'block_square.bmp', 'bmp');

block_square_small = imread('block_square_small.bmp');
block_square_small(block_square_small < 128) = 0;
block_square_small(block_square_small >= 128) = 255;
figure(1), imshow(block_square_small, 'border', 'tight')
unique(block_square_small)
imwrite(block_square_small, 'block_square_small.bmp', 'bmp');

block_star = imread('block_star.bmp');
block_star(block_star < 128) = 0;
block_star(block_star >= 128) = 255;
figure(1), imshow(block_star, 'border', 'tight')
unique(block_star)
imwrite(block_star, 'block_star.bmp', 'bmp');

block_star_small = imread('block_star_small.bmp');
block_star_small(block_star_small < 128) = 0;
block_star_small(block_star_small >= 128) = 255;
figure(1), imshow(block_star_small, 'border', 'tight')
unique(block_star_small)
imwrite(block_star_small, 'block_star_small.bmp', 'bmp');

block_text = imread('block_text.bmp');
block_text(block_text < 128) = 0;
block_text(block_text >= 128) = 255;
figure(1), imshow(block_text, 'border', 'tight')
unique(block_text)
imwrite(block_text, 'block_text.bmp', 'bmp');

block_triangle = imread('block_triangle.bmp');
block_triangle(block_triangle < 128) = 0;
block_triangle(block_triangle >= 128) = 255;
figure(1), imshow(block_triangle, 'border', 'tight')
unique(block_triangle)
imwrite(block_triangle, 'block_triangle.bmp', 'bmp');
