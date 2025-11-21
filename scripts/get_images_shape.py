import pathlib
import openslide
import argparse

def __main__() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='path/to/wsi/directory')
    parser.add_argument('--ext', type=str, help='file extension of the slides', default='.tif')
    parser.add_argument('--level', type=int, help='level to read the slide', default=0)
    parser.add_argument('--output', type=str, help='output file path', default='../images_shapes_lvl0/output.txt')
    args = parser.parse_args()

    # Create the output directory if it does not exist
    output_fpath = pathlib.Path(args.output)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(output_fpath, 'w') as out_file:
        path = pathlib.Path(args.datadir)
        slide_fpaths = [f for f in path.glob(f'*{args.ext}')]
        for fpath in slide_fpaths:
            slide = openslide.OpenSlide(str(fpath))
            width, height = slide.level_dimensions[args.level]
            print(f'{fpath.name}, {width}, {height}')
            out_file.write(f'{fpath.name},{width},{height}\n')
            slide.close()

if __name__ == '__main__':
    __main__()