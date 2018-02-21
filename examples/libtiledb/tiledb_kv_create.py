import tiledb

def main():

    ctx = tiledb.Ctx()

    # KV objects are limited to storing string keys/values for the time being
    a1 = tiledb.Attr(ctx, "value", compressor=("gzip", -1), dtype=bytes)
    kv = tiledb.KV(ctx, "my_kv", attrs=(a1,))

    # Dump the KV schema
    kv.dump()

    vals = {"key1": "a", "key2": "bb", "key3": "dddd"}
    print("Updating KV with values: {!r}\n".format(vals))

    # Update the KV
    kv.update(vals)

    # get kv item
    print("KV value for 'key3': {}\n".format(kv['key3']))

    # set kv item
    kv['key3'] = "eeeee"
    print("Updated KV value for 'key3': {}\n".format(kv['key3']))

    try:
        kv["don't exist"]
    except KeyError:
        print("KeyError was raised for key 'don't exist'\n")

    # consolidate kv updates
    kv.consolidate()

    # convert kv to Python dict
    kv_dict = dict(kv)
    print("Convert to Python dict: {!r}\n".format(kv_dict))


if __name__ == '__main__':
    main()
