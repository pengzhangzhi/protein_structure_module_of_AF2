import click
def main():
    ...
    
@click.command()
@click.option('--N', default = 2, help = 'len of amino acid')
def test(N):
    print("test structure module...",N)