from PIL import Image

def hide_logo(image_path, logo_path, output_path):
    # Open images
    image = Image.open(image_path)
    logo = Image.open(logo_path)
    
    # Ensure logo is not larger than the image
    if logo.size[0] > image.size[0] or logo.size[1] > image.size[1]:
        print("Logo is too large. Resizing it.")
        logo.thumbnail((image.size[0], image.size[1]))
    
    # Convert images to RGB
    image = image.convert('RGB')
    logo = logo.convert('RGB')
    
    # Get pixel data
    image_pixels = image.load()
    logo_pixels = logo.load()
    
    # Hide logo in the least significant bit
    for y in range(logo.size[1]):
        for x in range(logo.size[0]):
            # Get pixel values
            r, g, b = logo_pixels[x, y]
            
            # Convert pixel values to binary and remove the least significant bit
            r_bin = format(r, '08b')[:-1]
            g_bin = format(g, '08b')[:-1]
            b_bin = format(b, '08b')[:-1]
            
            # Get the corresponding pixel in the original image
            img_x, img_y = x, y
            if img_x >= image.size[0] or img_y >= image.size[1]:
                break
            
            # Get original pixel values
            img_r, img_g, img_b = image_pixels[img_x, img_y]
            
            # Replace the least significant bit of the original image's pixel with the logo's pixel
            # Using LSB, bit plane 0
            img_r = (img_r & ~1) | int(r_bin[-1])
            img_g = (img_g & ~1) | int(g_bin[-1])
            img_b = (img_b & ~1) | int(b_bin[-1])
            
            # For Using MSB you can replace above lines with this lines :
            # Replace LSB manipulation with MSB manipulation

            ## img_r = (img_r & 127) | (int(r_bin[0]) << 7)
            ## img_g = (img_g & 127) | (int(g_bin[0]) << 7)
            ## img_b = (img_b & 127) | (int(b_bin[0]) << 7)

            
            # Update the pixel in the original image
            image_pixels[img_x, img_y] = (img_r, img_g, img_b)
    
    # Save the modified image
    image.save(output_path)

# Example usage
hide_logo('img/Screenshot_7-4-2025_171929_.jpeg', 'Ex2\img\Nooshirvani_of_Babol_University_of_Technology_Logo.png', 'output_image.png')
