import plotly.express as px


def plotly_hist(df, x, des):
    """
    The main function to plot based on the input dataframe and its header

    Parameters
    ----------
    df: pandas dataframe
         the x axis data
    x: string or integer, optional (default=0)
       header or position of data in the dfx

    Returns
    -------
    plotly.figure.Figure object

    """
    # check data
    if isinstance(x, str):
        X = df[x].values
    elif isinstance(x, int):
        X = df.iloc[:, x].values
    else:
        msg = 'x must be string for the header or integer for the postion of data in the dfx'
        raise TypeError(msg)
    fig = px.histogram(df, x=x, title=des)
    return fig


def plotly_bar(df, x, y, des):
    """
    The main function to plot based on the input dataframe and its header

    Parameters
    ----------
    df: pandas dataframe
         the x axis data
    x: string or integer, optional (default=0)
       header or position of data in the dfx
    y: string or integer, optional (default=0)
       header or position of data in the dfy
    des: title for plot

    Returns
    -------
    plotly.figure.Figure object

    """
    # check data
    if isinstance(x, str):
        X = df[x].values
    elif isinstance(x, int):
        X = df.iloc[:, x].values
    else:
        msg = 'x must be string for the header or integer for the postion of data in the dfx'
        raise TypeError(msg)

    if isinstance(y, str):
        Y = df[y].values
    elif isinstance(y, int):
        Y = df.iloc[:, y].values
    else:
        msg = 'y must be string for the header or integer for the postion of data in the dfy'
        raise TypeError(msg)
    fig = px.bar(x=df[x].values, y=df[y].values, title=des)
    return fig


def plotly_scatter(df, x, y, z, des):
    """
    the main function to plot based on the input dataframes and their headers

    Parameters
    ----------
    df: pandas dataframe
         the x axis data
    x: string or integer, optional (default=0)
       header or position of data in the dfx
    y: string or integer, optional (default=0)
       header or position of data in the dfy
    des: title for plot

    Returns
    -------
    plotly.figure.Figure object

    """
    # check data
    if isinstance(x, str):
        X = df[x].values
    elif isinstance(x, int):
        X = df.iloc[:, x].values
    else:
        msg = 'x must be string for the header or integer for the postion of data in the dfx'
        raise TypeError(msg)

    if isinstance(y, str):
        Y = df[y].values
    elif isinstance(y, int):
        Y = df.iloc[:, y].values
    else:
        msg = 'y must be string for the header or integer for the postion of data in the dfy'
        raise TypeError(msg)

    # instantiate figure
    fig = px.scatter(df, x=df[x], y=df[y], title=des, color=z)
    # trash = ax.plot(X,Y,color=self.color, marker=self.marker, linestyle=self.linestyle, linewidth= self.linewidth)
    return fig


def plotly_line(df, x, y, des):
    """
    The main function to plot based on the input dataframe and its header

    Parameters
    ----------
    df: pandas dataframe
         the x axis data
    x: string or integer, optional (default=0)
       header or position of data in the dfx
    y: string or integer, optional (default=0)
       header or position of data in the dfy
    des: title for plot

    Returns
    -------
    plotly.figure.Figure object

    """
    # check data
    if isinstance(x, str):
        X = df[x].values
    elif isinstance(x, int):
        X = df.iloc[:, x].values
    else:
        msg = 'x must be string for the header or integer for the postion of data in the dfx'
        raise TypeError(msg)

    if isinstance(y, str):
        Y = df[y].values
    elif isinstance(y, int):
        Y = df.iloc[:, y].values
    else:
        msg = 'y must be string for the header or integer for the postion of data in the dfy'
        raise TypeError(msg)
    fig = px.line(df, title=des)
    return fig
