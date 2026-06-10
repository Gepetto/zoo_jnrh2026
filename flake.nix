{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          inputs',
          pkgs,
          self',
          ...
        }:
        {
          packages = {
            zoo-tuto = pkgs.python3Packages.callPackage ./package.nix { };
            default = pkgs.python3.withPackages (p: [ self'.packages.zoo-tuto ]);
            container = pkgs.dockerTools.buildImage {
              name = "gepetto/zoo-jnrh2026";
              tag = "latest";
              copyToRoot = [ self'.packages.default ];
              config = {
                Entrypoint = [
                  "/bin/jupyter"
                  "lab"
                  "--allow-root"
                ];
                WorkingDir = "/tuto";
              };
            };
          };
        };
    };
}
