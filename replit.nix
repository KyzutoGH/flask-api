{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.flask
    pkgs.python311Packages.tensorflow
    pkgs.python311Packages.numpy
    pkgs.python311Packages.scikit-learn
  ];
}