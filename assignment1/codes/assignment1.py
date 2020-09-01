{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "assignment1",
      "provenance": [],
      "authorship_tag": "ABX9TyNFGkoARquk7q7mxnn5MlBo",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/shreeprasadbhat/matrix-theory/blob/master/assignment1/codes/assignment1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wArRwBZcqViN",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "33379337-b179-461a-ee55-a70353965b96"
      },
      "source": [
        "import numpy as np\n",
        "\n",
        "a = np.array([3, -4, 5])\n",
        "b = np.array([-2, 1, -3])\n",
        "\n",
        "cross_product = np.cross(a, b)\n",
        "\n",
        "scalar_product = np.dot(a, b)\n",
        "\n",
        "print ('Cross product of a and b is '+ str(cross_product))\n",
        "\n",
        "print ('Scalar product of a and b is ' + str(scalar_product))"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cross product of a and b is [ 7 -1 -5]\n",
            "Scalar product of a and b is -25\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}