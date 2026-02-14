import newton.examples

try:
    from examples.example_minimal_usd_import import (
        DEFAULT_CONFIG,
        Example as ImportExample,
        _create_parser as _create_import_parser,
    )
except ModuleNotFoundError:
    from example_minimal_usd_import import (  # type: ignore[no-redef]
        DEFAULT_CONFIG,
        Example as ImportExample,
        _create_parser as _create_import_parser,
    )


def _create_parser():
    parser = _create_import_parser()
    parser.set_defaults(
        viewer="gl",
        output_path="output/minimal_usd_export.anim.usda",
        use_layered_usd=True,
        copy_usd=False,
    )
    return parser


class Example(ImportExample):
    def __init__(self, viewer, args, cfg=DEFAULT_CONFIG):
        super().__init__(viewer, args, cfg=cfg)
        self.frame_num = 0
        self._frame_attr = None
        self._init_frame_num_attr()

    def _init_frame_num_attr(self) -> None:
        stage = getattr(self.viewer, "stage", None)
        if stage is None:
            return
        try:
            from pxr import Sdf  # noqa: PLC0415
        except Exception:
            return

        root = getattr(self.viewer, "root", None)
        prim = root.GetPrim() if root is not None else stage.GetDefaultPrim()
        if prim is None or not prim.IsValid():
            return
        self._frame_attr = prim.CreateAttribute("frameNum", Sdf.ValueTypeNames.Int, custom=True)

    def _write_frame_num_attr(self) -> None:
        if self._frame_attr is None:
            return
        frame_index = int(getattr(self.viewer, "_frame_index", 0))
        self._frame_attr.Set(int(self.frame_num), frame_index)

    def render(self):
        super().render()
        self._write_frame_num_attr()
        self.frame_num += 1


def main():
    parser = _create_parser()
    viewer, args = newton.examples.init(parser)
    viewer_name = str(getattr(args, "viewer", "")).lower()
    if args.use_layered_usd and viewer_name == "usd":
        raise ValueError(
            "--use_layered_usd cannot be combined with --viewer usd in example_minimal_usd_export. "
            "Use --viewer gl or --viewer null."
        )

    example = Example(viewer, args, cfg=DEFAULT_CONFIG)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
