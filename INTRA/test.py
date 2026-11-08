#!/usr/bin/env python3

import sys


def main():
	with open("thetas.txt", "r") as f:
		content = f.read()
		print(content)

		parts = content.split(",")
		print(parts[0], file=sys.stderr)


if __name__ == "__main__":
	main()