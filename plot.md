# The plot (`pl`) accessor

The plotting accessor provides several plotting functions.

## `ds.pl` (PlotAccessor)

Adds plotting functions to the image container.

Available on any `xarray.Dataset` after `import spatialproteomics as sp`, as `ds.pl`.

### `ds.pl.colorize(colors: List[str]=['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', ...], background: str='black', normalize: bool=True, merge: bool=True, layer_key: str='_image', amin: Union[float, List, NoneType]=None, amax: Union[float, List, NoneType]=None) -> xr.Dataset`

Colorizes a stack of images.

**Parameters**

colors : List[str], optional
    A list of strings that denote the color of each channel.
background : str, optional
    Background color of the colorized image. Default is "black".
normalize : bool, optional
    Normalize the image prior to colorizing it. Default is True. When amin and amax are not specified, this uses the 3- and 99.8-percentile for normalization. If they are specified, it uses them as absolute values.
merge : True, optional
    Merge the channel dimension. Default is True.
layer_key : str, optional
    The key of the layer containing the image. Default is '_image'.
amin : float | List, optional
    The minimum absolute value for normalization (either global or per channel). Default is None.
amax : float | List, optional
    The maximum absolute value for normalization (either global or per channel). Default is None.

**Returns**

xr.Dataset
    The image container with the colorized image stored in '_plot'.

### `ds.pl.show(render_image: bool=True, render_segmentation: bool=False, render_labels: bool=False, render_neighborhoods: bool=False, ax=None, legend_image: bool=True, legend_segmentation: bool=True, legend_label: bool=True, legend_neighborhoods: bool=True, background: str='black', downsample: int=1, label_order: Optional[List]=None, neighborhood_order: Optional[List]=None, legend_kwargs: dict={'framealpha': 1}, segmentation_kwargs: dict={}, label_kwargs: dict={}, neighborhood_kwargs: dict={}) -> xr.Dataset`

Display an image with optional rendering elements. Can be used to render intensities, segmentation masks and labels, either individually or all at once.

**Parameters**

render_image : bool
    Whether to render the image with channel intensities. Default is True.
render_segmentation : bool
    Whether to render segmentation. Default is False.
render_labels : bool
    Whether to render labels. Default is False.
render_neighborhoods : bool
    Whether to render neighborhoods. Default is False.
ax
    The matplotlib axes to plot on. If None, the current axes will be used.
legend_image : bool
    Whether to show the channel legend. Default is True.
legend_segmentation : bool
    Whether to show the segmentation legend (only becomes relevant when dealing with multiple segmentation layers, e. g. when using cellpose). Default is False.
legend_label : bool
    Whether to show the label legend. Default is True.
legend_neighborhoods : bool
    Whether to show the neighborhood legend. Default is True.
background : str
    Background color of the image. Default is "black".
downsample : int
    Downsample factor for the image. Default is 1 (no downsampling).
label_order : list
    A list specifying the order of the label indices. Default is None.
neighborhood_order : list
    A list specifying the order of the neighborhood indices. Default is None.
legend_kwargs : dict
    Keyword arguments for configuring the legend. Default is {"framealpha": 1}.
segmentation_kwargs : dict
    Keyword arguments for rendering the segmentation. Default is {}.
label_kwargs : dict
    Keyword arguments for rendering the labels. Default is {}.
neighborhood_kwargs : dict
    Keyword arguments for rendering the neighborhoods. Default is {}.

**Returns**

xr.Dataset
    The dataset object including the plot.

**Raises**

- AssertionError: If no rendering element is specified.

**Notes**

- This method displays an image with optional rendering elements such as intensities, labels, and segmentation.
- The `render_intensities`, `render_labels`, and `render_segmentation` parameters control which rendering elements are displayed.
- The `ax` parameter allows specifying a specific matplotlib axes to plot on. If None, the current axes will be used.
- The `show_channel_legend` and `show_label_legend` parameters control whether to show the channel and label legends, respectively.
- The `legend_kwargs`, `segmentation_kwargs`, and `label_kwargs` parameters allow configuring the appearance of the legend, segmentation, and labels, respectively.

### `ds.pl.annotate(variable: str='cell', layer_key: str='_obs', highlight: list=[], text_kwargs: dict={'color': 'w', 'fontsize': 12}, highlight_kwargs: dict={'color': 'w', 'fontsize': 16, 'fontweight': 'bold'}, bbox: Optional[List]=None, format_string: str='', ax=None) -> xr.Dataset`

Annotates cells with their respective number on the plot.

**Parameters**

variable : str, optional
    The feature in the observation table to be used for cell annotation. Default is "cell".
layer_key : str, optional
    The key representing the layer in the data object. Default is Layers.OBS.
highlight : list, optional
    A list containing cell IDs to be highlighted in the plot.
text_kwargs : dict, optional
    Keyword arguments passed to matplotlib's text function for normal cell annotations.
highlight_kwargs : dict, optional
    Similar to 'text_kwargs' but specifically for the cell IDs passed via 'highlight'.
bbox : Union[List, None], optional
    A list containing bounding box coordinates [xmin, xmax, ymin, ymax] to annotate cells only within the box.
    Default is None, which annotates all cells.
format_string : str, optional
    The format string used to format the cell annotation text. Default is "" (no formatting).
ax : matplotlib.axes, optional
    The matplotlib axis to plot on. If not provided, the current axis will be used.

**Returns**

xr.Dataset
    The updated image container.

**Notes**

- The function annotates cells with their respective values from the selected feature.
- You can highlight specific cells in the plot using the 'highlight' parameter.
- Bounding box coordinates can be provided via 'bbox' to annotate cells only within the specified box.
- 'format_string' can be used to format the cell annotation text (e.g., "{t:.2f}" for float formatting).

### `ds.pl.render_segmentation(layer_key: str='_segmentation', colors: List[str]=['white', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', ' ...], alpha: float=0.0, alpha_boundary: float=1.0, mode: str='inner') -> xr.Dataset`

Renders the segmentation mask with optional alpha blending and boundary rendering.

Parameters:
    layer_key (str, optional): The key of the layer containing the segmentation mask. Default is '_segmentation'.
    colors (List[str], optional): A list of colors to be used for rendering the segmentation mask. Default is ['white'].
    alpha (float, optional): The alpha value for blending the segmentation mask with the plot. Default is 0.0.
    alpha_boundary (float, optional): The alpha value for rendering the boundary of the segmentation mask. Default is 1.0.
    mode (str, optional): The mode for rendering the segmentation mask. Possible values are "inner" and "outer". Default is "inner".

Returns:
    xr.Dataset: The modified xarray Dataset.

Raises:
    AssertionError: If no segmentation layer is found in the object.

Note:
    - The segmentation mask must be added to the object before calling this method.
    - The segmentation mask is expected to have a single channel with integer labels.
    - The rendered segmentation mask will be added as a new layer to the object.

### `ds.pl.render_labels(alpha: float=1.0, alpha_boundary: float=1.0, mode: str='inner', override_color: Optional[str]=None) -> xr.Dataset`

Renders cell type labels on the plot.

Parameters:
    alpha (float, optional): The transparency of the labels. Defaults to 1.0.
    alpha_boundary (float, optional): The transparency of the label boundaries. Defaults to 1.0.
    mode (str, optional): The mode of rendering. Can be "inner" or "outer". Defaults to "inner".
    override_color (str, optional): The color to override the label colors. Defaults to None.

Returns:
    xr.Dataset: The modified dataset with rendered labels.

Raises:
    AssertionError: If no labels are found in the object.

Notes:
    - This method requires labels to be present in the object. Add labels first using `la.predict_cell_types_argmax()` or `tl.astir()`.
    - The `mode` parameter determines whether the labels are rendered inside or outside the label boundaries.
    - The `override_color` parameter can be used to override the label colors with a single color.

### `ds.pl.render_neighborhoods(style: str='neighborhoods', alpha: float=1.0, alpha_boundary: float=1.0, boundary_color: str='dimgray', boundary_thickness: int=3, dilation_strength: int=40, erosion_strength: int=35) -> xr.Dataset`

Render neighborhoods on the spatial data.

**Parameters**

style : str, optional
    The style of rendering, either 'cells' or 'neighborhoods'. Default is 'neighborhoods'.
alpha : float, optional
    The alpha transparency for the rendered image. Default is 1.0.
alpha_boundary : float, optional
    The alpha transparency for the boundary of the rendered image. Default is 1.0.
boundary_color : str, optional
    The color of the boundary lines. Default is 'dimgray'.
boundary_thickness : int, optional
    The thickness of the boundary lines. Default is 3.
dilation_strength : int, optional
    The strength of the dilation applied to the cells. Default is 40.
erosion_strength : int, optional
    The strength of the erosion applied to the cells. Default is 35.

**Returns**

xr.Dataset
    The dataset with the rendered neighborhoods.

**Raises**

AssertionError
    If no neighborhoods are found in the object or if the style is not 'cells' or 'neighborhoods'.

### `ds.pl.render_obs(feature: str=None, cmap: str='viridis', background: str='black', alpha: float=1.0, alpha_boundary: float=1.0, mode: str='inner', vmin: Optional[float]=None, vmax: Optional[float]=None) -> xr.Dataset`

Render the observation layer with the specified feature and colormap.

**Parameters**

feature : str, optional
    The feature to be rendered from the observation layer. Default is None.
cmap : str, optional
    The colormap to be used for rendering. Default is "viridis".
background : str, optional
    The background color of the rendered image. Default is "black".
alpha : float, optional
    The alpha value for the rendered image. Default is 1.0.
alpha_boundary : float, optional
    The alpha value for the boundaries in the rendered image. Default is 1.0.
mode : str, optional
    The mode for rendering. Default is "inner".
vmin : float, optional
    The minimum value for colormap normalization. Default is None.
vmax : float, optional
    The maximum value for colormap normalization. Default is None.

**Returns**

xr.Dataset
    The dataset with the rendered observation layer.

**Raises**

AssertionError
    If the observation layer or segmentation layer is not found in the object.
    If the specified feature is not found in the observation layer.

### `ds.pl.imshow(legend_image: bool=False, legend_segmentation: bool=False, legend_label: bool=False, legend_neighborhoods: bool=False, legend_obs: bool=False, downsample: int=1, label_order: Optional[List[str]]=None, neighborhood_order: Optional[List[str]]=None, legend_kwargs: dict={'framealpha': 1}, cbar_kwargs: dict={}, ax=None)`

Plots the image after rendering certain layers.
Meant to be used in conjunction with pl.colorize(), pl.render_segmentation, pl.render_label() or pl.render_neighborhoods().
For a more high level wrapper, please refer to pl.show() instead.

**Parameters**

legend_image : bool, optional
    Show the legend for the channels. Default is False.
legend_segmentation : bool, optional
    Show the legend for the segmentation. Default is False.
legend_label : bool, optional
    Show the label legend. Default is False.
legend_neighborhoods : bool, optional
    Show the neighborhood legend. Default is False.
legend_obs: bool, optional
    Show the observation colorbar. Default is False.
downsample : int, optional
    Downsample factor for the image. Default is 1.
legend_kwargs : dict, optional
    Additional keyword arguments for configuring the legend. Default is {"framealpha": 1}.
cbar_kwargs : dict, optional
    Additional keyword arguments for configuring the colorbar. Default is {}.
label_order: Optional[List[str]], optional
    The order in which the labels should be displayed. Default is None.
neighborhood_order: Optional[List[str]], optional
    The order in which the neighborhoods should be displayed. Default is None.
ax : matplotlib.axes, optional
    The matplotlib axis to plot on. If not provided, the current axis will be used.

**Returns**

xr.Dataset
    The updated image container.

**Notes**

- The function is used to plot images in conjunction with 'im.colorize' and 'la.render_label'.
- The appearance of the plot and the inclusion of legends can be controlled using the respective parameters.

### `ds.pl.scatter_labels(legend: bool=True, size: float=1.0, alpha: float=0.9, zorder: int=10, render_edges: bool=False, render_self_edges: bool=False, ax=None, legend_kwargs: dict={'framealpha': 1}, scatter_kwargs: dict={}) -> xr.Dataset`

Scatter plot of labeled cells.

**Parameters**

legend : bool, optional
    Whether to show the legend. Default is True.
size : float, optional
    Size of the scatter markers. Default is 1.0.
alpha : float, optional
    Transparency of the scatter markers. Default is 0.9.
zorder : int, optional
    The z-order of the scatter markers. Default is 10.
render_edges : bool, optional
    Whether to render the edges between cells within the same neighborhood. Default is False.
render_self_edges : bool, optional
    Whether to render the edges between cells and themselves. Default is False.
ax : matplotlib.axes.Axes, optional
    The axes on which to plot the scatter. If not provided, the current axes will be used.
legend_kwargs : dict, optional
    Additional keyword arguments for configuring the legend. Default is {"framealpha": 1}.
scatter_kwargs : dict, optional
    Additional keyword arguments for configuring the scatter plot.

**Returns**

xr.Dataset
    The modified spatialproteomics object.

### `ds.pl.scatter(feature: str, palette: dict=None, legend: bool=True, layer_key: str='_obs', size: float=1.0, alpha: float=0.9, zorder: int=10, ax=None, legend_kwargs: dict={'framealpha': 1}, scatter_kws: dict={}) -> xr.Dataset`

Create a scatter plot of some feature. At the moment, only categorical features are supported.

**Parameters**

feature : str
    The feature to be plotted.
palette : dict, optional
    A dictionary mapping feature values to colors. If not provided, a default palette will be used.
legend : bool, optional
    Whether to show the legend. Default is True.
layer_key : str, optional
    The key of the layer to be plotted. Default is Layers.OBS.
size : float, optional
    The size of the scatter points. Default is 1.0.
alpha : float, optional
    The transparency of the scatter points. Default is 0.9.
zorder : int, optional
    The z-order of the scatter points. Default is 10.
ax : object, optional
    The matplotlib axes object to plot on. If not provided, the current axes will be used.
legend_kwargs : dict, optional
    Additional keyword arguments for configuring the legend. Default is {"framealpha": 1}.
scatter_kws : dict, optional
    Additional keyword arguments for configuring the scatter plot. Default is {}.

**Returns**

xr.Dataset
    The original data object.

**Raises**

- AssertionError: If the layer_key is not found in the data object.
- AssertionError: If the feature is not found in the specified layer.
- AssertionError: If the X or Y coordinates are not found in the specified layer.
- AssertionError: If the number of unique feature values is greater than 10 and no color_scheme is provided.
- AssertionError: If not all unique feature values are present in the provided palette.

### `ds.pl.add_box(xlim: List[int], ylim: List[int], color: str='w', linewidth: float=2, ax=None)`

Adds a rectangular box to the current plot.

**Parameters**

xlim : List[int]
    The x-bounds of the box [xstart, xstop].
ylim : List[int]
    The y-bounds of the box [ymin, ymax].
color : str, optional
    The color of the box. Default is "w" (white).
linewidth : float, optional
    The linewidth of the box. Default is 2.
ax : matplotlib.axes, optional
    The matplotlib axis to plot on. If not provided, the current axis will be used.

**Returns**

xr.Dataset
    The updated spatialproteomics object.

### `ds.pl.autocrop(padding: int=50, downsample: int=10, key: str='_image', channel: Optional[str]=None) -> xr.Dataset`

Crop the image so that the background surrounding the tissue/TMA gets cropped away.

**Parameters**

padding : int
    The padding to be added around the cropped image in pixels. Default is 50.
downsample : int
    The downsampling factor for the image. Default is 10.
key : str
    The key of the image to be cropped. Default is Layers.IMAGE.
channel : str, optional
    The channel used for cropping. Default is None, which defaults to using the first available channel.

**Returns**

xr.Dataset
    The cropped image.
